
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
import os
import argparse
import datetime
import time
import sys
sys.path.insert(0, './src')
import utils
import iwae1
import iwae2
import DMix
from data_hand import *
from keras.utils import to_categorical
from Utils2 import *
from utils import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# TODO: control warm-up from commandline
parser = argparse.ArgumentParser()
parser.add_argument("--stochastic_layers", type=int, default=1, choices=[1, 2], help="number of stochastic layers in the model")
parser.add_argument("--n_samples", type=int, default=50, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--objective", type=str, default="vae_elbo", choices=["vae_elbo", "iwae_elbo", "iwae_eq14", "vae_elbo_kl"])
parser.add_argument("--gpu", type=str, default='0', help="Choose GPU")
args = parser.parse_args()
print(args)

# ---- string describing the experiment, to use in tensorboard and plots
string = "main_{0}_{1}_{2}".format(args.objective, args.stochastic_layers, args.n_samples)

'''
# ---- set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ---- dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ---- number of passes over the data, see bottom of page 6 in [1]
'''

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255

Xtest = utils.bernoullisample(Xtest)

#----load the second dataset
(Xtrain2, ytrain2), (Xtest2, ytest2) = keras.datasets.fashion_mnist.load_data()
Ntrain2 = Xtrain2.shape[0]
Ntest2 = Xtest2.shape[0]

# ---- reshape to vectors
Xtrain2 = Xtrain2.reshape(Ntrain2, -1) / 255
Xtest2 = Xtest2.reshape(Ntest2, -1) / 255

Xtest2 = utils.bernoullisample(Xtest2)
y_train2 = to_categorical(ytrain2, num_classes=10)
ytest2 = to_categorical(ytest2, num_classes=10)

# ---- do the training
start = time.time()
best = float(-np.inf)

#Split MNIST into Five tasks
y_train = to_categorical(ytrain, num_classes=10)
ytest = to_categorical(ytest, num_classes=10)

omnistTrainingSet,omnistTestingSet = Load_OMNIST(True)
omnistTestingSet = utils.bernoullisample(omnistTestingSet)

totalSet = np.concatenate((Xtrain,Xtrain2,omnistTrainingSet),axis=0)
totalTestingSet = np.concatenate((Xtest,Xtest2,omnistTestingSet),axis=0)

print(np.shape(totalSet))

taskCount = 5

totalResults = []

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 50
        self.len_discrete_code = 4
        self.epoch = 50

        self.learning_rate = 0.0001
        self.beta1 = 0.5

        self.beta = 0.1
        self.data_dim = 28*28

        self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.data_dim])
        self.text_k = tf.tile(self.input_x,[5000, 1])

        self.NofImportanceSamples = 50

        self.latentZArr = []
        self.latentXArr = []

        self.testLatentZArr = []
        self.testLatentXArr = []

        self.lossArr = []
        self.recoArr = []
        self.testLossArr = []
        self.KlArr = []
        self.TestKLArr = []

    def shoaared_encoder(self,name, x, z_dim=20, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
            mu = tf.layers.dense(l1, z_dim, activation=None)
            sigma = tf.layers.dense(l1, z_dim, activation=tf.exp)
            return mu, sigma

    def encoder(self,name, x, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            mu = tf.layers.dense(l1, z_dim, activation=None)
            sigma = tf.layers.dense(l1, z_dim, activation=tf.exp)
            return mu, sigma

    def shared_decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.tanh)
            return l1

    def decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.relu)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            x_hat = tf.layers.dense(
                l1, self.data_dim, activation=None)
            return x_hat

    def logmeanexp(self,log_w, axis):
        max = tf.reduce_max(log_w, axis=axis)
        return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max

    def Create_Component(self,index):
        if np.shape(self.lossArr)[0] == 0:
            sharedEncoderName = "sharedEncoder"
            encoderName = "Encoder" + str(index)
            sharedDecoderName = "sharedDecoder"
            decoderName = "Decoder" + str(index)
            x_k = self.input_x
            q_mu_shared, q_std_shared = self.shoaared_encoder(sharedEncoderName, x_k,self.z_dim, reuse=False)
            qzx_shared = tfd.Normal(q_mu_shared, q_std_shared + 1e-6)
            z_shared = qzx_shared.sample(1)
            z_shared = tf.reshape(z_shared,(self.batch_size,-1))

            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=False)

            n_samples = self.NofImportanceSamples
            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)

            self.latentZArr.append(z)

            x_shared = self.shared_decoder(sharedDecoderName, z, self.z_dim, reuse=False)

            self.latentXArr.append(x_shared)

            logits = self.decoder(decoderName, x_shared, self.z_dim, reuse=False)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)

            beta = 1.0
            log_w = lpxz + beta * (lpz - lqzx)

            kl = (lpz - lqzx)
            self.KlArr.append(kl)

            # mean over samples and batch
            vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
            vae_elbo_kl = tf.reduce_mean(lpxz) - beta * tf.reduce_mean(kl)

            # ---- IWAE elbos
            # eq (8): logmeanexp over samples and mean over batch
            iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
            trainingloss = -iwae_elbo

            self.lossArr.append(trainingloss)
            self.vaeLoss = trainingloss

            #testing loss
            n_samples = 5000
            z = qzx.sample(n_samples)
            self.testLatentZArr.append(z)

            x_shared = self.shared_decoder(sharedDecoderName, z, self.z_dim, reuse=True)
            self.testLatentXArr.append(x_shared)

            logits = self.decoder(decoderName, x_shared, self.z_dim, reuse=True)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)

            kl = (lpz - lqzx)
            self.TestKLArr.append(kl)

            beta = 1.0
            log_w = lpxz + beta * (lpz - lqzx)
            test_iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)

            self.testLossArr.append(test_iwae_elbo)
            #end of the test loss

            T_vars = tf.trainable_variables()
            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(trainingloss, var_list=T_vars)

        else:
            sharedEncoderName = "sharedEncoder"
            encoderName = "Encoder" + str(index)
            sharedDecoderName = "sharedDecoder"
            decoderName = "Decoder" + str(index)
            EncoderWeight = "EncoderWeight"+ str(index)
            DecoderWeight = "DecoderWeight"+ str(index)

            x_k = self.input_x
            q_mu_shared, q_std_shared = self.shoaared_encoder(sharedEncoderName, x_k, self.z_dim, reuse=True)
            qzx_shared = tfd.Normal(q_mu_shared, q_std_shared + 1e-6)
            z_shared = qzx_shared.sample(1)
            z_shared = tf.reshape(z_shared, (self.batch_size, -1))

            q_mu, q_std = self.encoder(encoderName, z_shared, self.z_dim, reuse=False)

            n_samples = self.NofImportanceSamples
            qzx = tfd.Normal(q_mu, q_std + 1e-6)
            z = qzx.sample(n_samples)

            # Create the component weight
            b_init = tf.glorot_uniform_initializer()
            # 1st hidden layer
            b0 = tf.get_variable(EncoderWeight, [np.shape(self.latentZArr)[0] + 1], initializer=b_init)
            b0 = tf.nn.softmax(b0)

            sumZ = z * b0[0]
            for i in range(np.shape(self.latentZArr)[0]):
                sumZ = sumZ + self.latentZArr[i] * b0[i + 1]

            self.latentZArr.append(sumZ)
            latentX1 = self.shared_decoder(sharedDecoderName, sumZ, z_dim=50, reuse=True)
            decoderWeight = tf.get_variable(DecoderWeight, [np.shape(self.latentXArr)[0] + 1], initializer=b_init)
            decoderWeight = tf.nn.softmax(decoderWeight)

            sumZ_genertor = latentX1 * decoderWeight[0]
            for i in range(np.shape(self.latentXArr)[0]):
                sumZ_genertor = sumZ_genertor + self.latentXArr[i] * decoderWeight[i + 1]

            self.latentXArr.append(sumZ_genertor)

            logits = self.decoder(decoderName, sumZ_genertor, z_dim=50, reuse=False)

            pxz = tfd.Bernoulli(logits=logits)

            pz = tfd.Normal(0, 1)

            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

            lpxz = tf.reduce_sum(pxz.log_prob(self.input_x), axis=-1)
            KLsum = 0
            for i in range(np.shape(self.lossArr)[0]):
                kl1 = b0[i+1] * self.KlArr[i]
                KLsum = KLsum + kl1

            beta = 1.0
            kl = (lpz - lqzx) * b0[0]
            KLsum = KLsum + kl

            self.KlArr.append(KLsum)

            log_w = lpxz + beta * KLsum
            iwae_elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
            trainingloss = -iwae_elbo
            self.lossArr.append(trainingloss)

            #testing loss
            qzx_test = tfd.Normal(q_mu, q_std + 1e-6)
            z_test = qzx.sample(5000)

            sumZ_test = z_test * b0[0]
            for i in range(np.shape(self.latentZArr)[0] - 1):
                sumZ_test = sumZ_test + self.testLatentZArr[i] * b0[i + 1]

            self.testLatentZArr.append(sumZ_test)

            latentX1_test = self.shared_decoder(sharedDecoderName, sumZ_test, z_dim=50, reuse=True)

            sumZ_genertor_test = latentX1_test * decoderWeight[0]
            for i in range(np.shape(self.latentXArr)[0] - 1):
                sumZ_genertor_test = sumZ_genertor_test + self.testLatentXArr[i] * decoderWeight[i + 1]

            self.testLatentXArr.append(sumZ_genertor_test)
            logits_test = self.decoder(decoderName, sumZ_genertor_test, z_dim=50, reuse=True)

            pxz_test = tfd.Bernoulli(logits=logits_test)

            pz_test = tfd.Normal(0, 1)

            lpz_test = tf.reduce_sum(pz_test.log_prob(z_test), axis=-1)

            lqzx_test = tf.reduce_sum(qzx_test.log_prob(z_test), axis=-1)

            lpxz_test = tf.reduce_sum(pxz_test.log_prob(self.input_x), axis=-1)
            KLsum = 0
            for i in range(np.shape(self.lossArr)[0] - 1):
                kl1 = b0[i + 1] * self.TestKLArr[i]
                KLsum = KLsum + kl1

            beta = 1.0
            kl = (lpz_test - lqzx_test) * b0[0]
            KLsum = KLsum + kl

            self.TestKLArr.append(KLsum)

            log_w_test = lpxz_test + beta * KLsum
            iwae_elbo_test = tf.reduce_mean(self.logmeanexp(log_w_test, axis=0), axis=-1)
            self.testLossArr.append(iwae_elbo_test)
            #end of testing loss

            T_vars = tf.trainable_variables()
            vars1 = [var for var in T_vars if var.name.startswith(decoderName)]
            vars2 = [var for var in T_vars if var.name.startswith(encoderName)]
            vars3 = [var for var in T_vars if var.name.startswith(EncoderWeight)]
            vars4 = [var for var in T_vars if var.name.startswith(DecoderWeight)]

            vars = vars1 + vars2 + vars3 + vars4

            self.vaeLoss = trainingloss
            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(trainingloss, var_list=vars)

            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def Select_Component(self,test):
        myCount = int(np.shape(test)[0] / self.batch_size)

        lossArr = []
        for index in range(np.shape(self.lossArr)[0]):
            sumLoss = 0
            for i in range(myCount):
                batch = test[i * self.batch_size : (i + 1) * self.batch_size]
                loss = self.sess.run(self.lossArr[index],feed_dict={self.input_x:batch})
                sumLoss = sumLoss + loss
            sumLoss = sumLoss / myCount
            lossArr.append(sumLoss)

        minIndex = np.argmin(lossArr)
        return minIndex

    def Evaluation(self,test,index):
        mycount = int(np.shape(test)[0] / self.batch_size)
        sumLoss = 0
        for i in range(mycount):
            batch = test[i * self.batch_size: (i + 1) * self.batch_size]
            loss = self.sess.run(self.testLossArr[index], feed_dict={self.input_x: batch})
            sumLoss = sumLoss + loss
        sumLoss = sumLoss / mycount
        return sumLoss

    def EvaluationAndIndex(self,test):
        mycount = int(np.shape(test)[0] / self.batch_size)
        sumLoss = 0
        for i in range(mycount):
            batch = test[i * self.batch_size: (i + 1) * self.batch_size]
            index = self.Select_Component(batch)
            loss = self.sess.run(self.testLossArr[index], feed_dict={self.input_x: batch})
            sumLoss = sumLoss + loss
        sumLoss = sumLoss / mycount
        return sumLoss,index

    def Build(self):
        #Build the first component

        self.Create_Component(1)
        #self.Create_Component(2)

    def Train(self):
        pz = tfd.Normal(0, 1)
        step = 0
        taskCount = 1

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        self.totalSet = totalSet
        self.totalSet = np.array(self.totalSet)
        self.totalSet = np.reshape(self.totalSet,(-1,28*28))

        self.DynamicMmeory =self.totalSet[0:self.batch_size]
        self.maxMmeorySize = 512

        self.DynamicMmeory = np.array(self.DynamicMmeory)
        totalCount = int(np.shape(self.totalSet)[0] / self.batch_size)

        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            for index in range(totalCount):
                batch = self.totalSet[index * self.batch_size : (index + 1) * self.batch_size]
                epochs = 100
                currentX = Xtrain

                #self.Create_Component(2)
                for epoch in range(epochs):
                    # ---- binarize the training data at the start of each epoch
                    batch_binarized = utils.bernoullisample(batch)
                    Xtrain_binarized = utils.bernoullisample(self.DynamicMmeory)

                    #Evaluate the novelty of a new batch of samples
                    if np.shape(self.DynamicMmeory)[0] > self.maxMmeorySize:
                        oldLoss = 0
                        count = int(np.shape(Xtrain_binarized)[0] / self.batch_size)
                        for i in range(count):
                            bb = Xtrain_binarized[i*self.batch_size:(i+1)*self.batch_size]
                            loss = self.sess.run(self.lossArr[np.shape([self.lossArr])[0] - 1],feed_dict={self.input_x:bb} )
                            oldLoss = oldLoss + loss
                        oldLoss = oldLoss / count
                        newLoss = self.sess.run(self.lossArr[np.shape([self.lossArr])[0] - 1],feed_dict={self.input_x:batch_binarized})
                        threshold = 30 + 20
                        diff = np.abs(oldLoss - newLoss)
                        if diff > threshold:
                            if np.shape(self.lossArr)[0] < 30:
                                newindex = np.shape(self.lossArr)[0] + 1
                                print(newindex)
                                self.Create_Component(newindex)
                                self.DynamicMmeory = batch
                        else:
                            n_examples = np.shape(self.DynamicMmeory)[0]
                            index = [i for i in range(n_examples)]
                            np.random.shuffle(index)
                            self.DynamicMmeory = self.DynamicMmeory[index]
                            self.DynamicMmeory = self.DynamicMmeory[0:np.shape(self.DynamicMmeory)[0] - self.batch_size]
                            self.DynamicMmeory = np.concatenate((self.DynamicMmeory, batch), axis=0)
                    else:
                        self.DynamicMmeory = np.concatenate((self.DynamicMmeory, batch), axis=0)

                    Xtrain_binarized = utils.bernoullisample(self.DynamicMmeory)
                    n_examples = np.shape(Xtrain_binarized)[0]
                    index = [i for i in range(n_examples)]
                    np.random.shuffle(index)
                    Xtrain_binarized = Xtrain_binarized[index]
                    counter = 0

                    myCount = int(np.shape(Xtrain_binarized)[0]/self.batch_size)

                    for idx in range(myCount):
                        step = step + 1
                        step = step %100000

                        batchImages = Xtrain_binarized[idx*self.batch_size:(idx+1)*self.batch_size]
                        beta = 1.0
                        _, d_loss = self.sess.run([self.vae_optim, self.vaeLoss],
                                                  feed_dict={self.input_x: batchImages})

                        if step % 1 == 0:
                            # ---- monitor the test-set
                            L = 5000
                            #test_res = model.val_step(Xtest, n_samples, beta)

                            #print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                            #      .format(epoch, epochs, 0, total_steps, res[objective].numpy(), test_res[objective], took))
                            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                                  .format(epoch, epochs, 0, 0, d_loss, np.shape(self.lossArr)[0], 0))

            loss, _ = self.EvaluationAndIndex(totalTestingSet)
            print(loss)

RunCount = 5

model = LifeLone_MNIST()
model.Build()
model.Train()

'''
# ---- save final weights
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set


for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))
'''