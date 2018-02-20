import waves.viewer as viewer
import tensorflow as tf
import numpy as np


class TFWaveSolver:
    def __init__(self):
        self.N = 600

        self.sess = tf.InteractiveSession()

        # Initial Conditions -- some rain drops hit a pond

        # Set everything to zero
        self.u_init = np.zeros([self.N, self.N], dtype=np.float32)
        self.ut_init = np.zeros([self.N, self.N], dtype=np.float32)

        # Some rain drops hit a pond at random points
        for _ in range(40):
            a,b = np.random.randint(0, self.N, 2)
            self.u_init[a,b] = np.random.uniform()
        
        # Parameters:
        # eps -- time resolution
        # damping -- wave damping
        self.eps = tf.placeholder(tf.float32, shape=())
        self.damping = tf.placeholder(tf.float32, shape=())

        # Create variables for simulation state
        self.U  = tf.Variable(self.u_init)
        self.Ut = tf.Variable(self.ut_init)

        # Discretized PDE update rules
        self.U_ = self.U + self.eps * self.Ut
        self.Ut_ = self.Ut + self.eps * (self._laplace(self.U) + self.damping * self.Ut)

        # Initialize state to initial conditions
        tf.global_variables_initializer().run()

        # Operation to update the state
        self.step = tf.group(self.U.assign(self.U_), self.Ut.assign(self.Ut_))
    
    def update(self, frame):
        # Step simulation
        self.step.run({self.eps: 0.03, self.damping: 0.04})

    def _make_kernel(self, a):
        """Transform a 2D array into a convolution kernel"""
        a = np.asarray(a)
        a = a.reshape(list(a.shape) + [1,1])
        return tf.constant(a, dtype=1)
    
    def _simple_conv(self, x, k):
        """A simplified 2D convolution operation"""
        x = tf.expand_dims(tf.expand_dims(x, 0), -1)
        y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
        return y[0, :, :, 0]

    def _laplace(self, x):
        """Compute the 2D laplacian of an array"""
        laplace_k = self._make_kernel([[0.5, 1.0, 0.5],
                                [1.0, -6., 1.0],
                                [0.5, 1.0, 0.5]])
        return self._simple_conv(x, laplace_k)
    
    @property
    def Z(self):
        return self.U.eval()
    
    @property
    def X(self):
        return np.arange(0, self.N)
    
    @property
    def Y(self):
        return np.arange(0, self.N)


viewer.Viewer(TFWaveSolver()).start()