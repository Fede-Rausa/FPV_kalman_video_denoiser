import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve2d


class KalmanFilter:
    """
    Kalman Filter with adjustable parameters
    X0: state vector at time 0 of dim n : will evolve over time
    P0: initial covariance matrix of dim n x n : will evolve over time
    F: state transition matrix of dim n x n
    H: measurement matrix of dim m x n
    Q: process noise covariance matrix of dim n x n
    R: measurement noise covariance matrix of dim m x m
    Z: measurement vector (dimension less or equal to state vector) of dim m
    """
    def __init__(self, X0, P0, F, H, Q, R, save_history=False):
        X0 = np.array(X0).reshape(-1, 1)
        P0 = np.array(P0)
        self.X0 = X0  # Initial state vector
        self.P0 = P0  # Initial covariance matrix
        self.X = X0  # State vector
        self.P = P0  # Covariance matrix
        self.F = F   # State transition matrix
        self.H = H   # Measurement matrix
        self.Q = Q   # Process noise covariance
        self.R = R   # Measurement noise covariance
        self.save_history = save_history

        self.states = np.array([X0])
        self.pred_states = np.array([self.X])
        self.covariances = np.array([P0])
        self.pred_covariances = np.array([self.P])

        self.observed_mask = np.nonzero(np.any(self.H != 0, axis=0))[0]
        #print(self.observed_mask)
        self.nobserved = len(self.observed_mask)
        self.unobserved_mask = list(np.nonzero(~np.any(self.H != 0, axis=0))[0])
        self.nunobserved = len(self.unobserved_mask)
        self.statedim = self.X.shape[0]


    def reset(self, X0=None, P0=None):
        """Reset the filter state to initial conditions"""
        if X0 is not None:
            self.X0 = X0
        if P0 is not None:
            self.P0 = P0

        self.X = self.X0
        self.P = self.P0
        
        # self.states = np.array([self.X])
        # self.pred_states = np.array([self.X])
        # self.covariances = np.array([self.P])


    def predict(self):
        """Prediction step"""
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.save_history:
            self.pred_states=np.append(self.pred_states, [self.X], axis=0)
            self.pred_covariances=np.append(self.pred_covariances, [self.P], axis=0)
        return self.X.flatten()
    

    def update(self, Z):
        """Update step with measurement"""
        Z = np.array(Z).reshape(self.nobserved, 1)
        self.X = np.array(self.X).reshape(self.statedim, 1)

        # Innovation (residual)
        Y = Z - self.H @ self.X
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        

        # Update state estimate
        self.X = self.X + K @ Y
        
        # Update covariance estimate
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        if self.save_history:
            self.states=np.append(self.states, [self.X], axis=0)
            self.covariances=np.append(self.covariances, [self.P], axis=0)
        return self.X.flatten()


    def get_state_observed(self):
        #print('mask shape: '+str(self.observed_mask.shape))
        #print('X shape: '+str(self.X.shape))
        return self.X[self.observed_mask]

    def get_state_unobserved(self):
        return self.X[self.unobserved_mask]










class big_simple_kalman:
    def __init__(self, X0, P0=1, Q_var=3, R_var=10):

        '''
        A simple Kalman filter for large state vectors.
        It assumes a simple model with no covariance between the state variables,
        where all the state variables are observed, with equal measurement and process variance. 
        In this way, the Kalman gain can be computed as a scalar, 
        which makes the computation feasible for hundreds of thousands of variables.
        This is suitable for denoising applications where the state is large (e.g., image pixels).
        X0: initial state vector
        P0: initial estimate covariance (scalar)
        Q_var: process noise variance (scalar)
        R_var: measurement noise variance (scalar)'''

        self.X = X0
        self.P = P0
        self.Q = Q_var
        self.R = R_var
        

    def update(self, Z):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.X = self.X + K * (Z - self.X)
        self.P = (1 - K) * self.P
        return self.X.flatten()



class big_simple_steady_kalman():
    def __init__(self, X0, Q_var=3, R_var=10):

        '''
        A faster implementation of big_simple_kalman. 

        A simple Kalman filter for large state vectors.
        It assumes a simple model with no covariance between the state variables,
        where all the state variables are observed, with equal measurement and process variance. 
        In this way, the Kalman gain can be computed as a scalar, 
        which makes the computation feasible for hundreds of thousands of variables.
        This is suitable for denoising applications where the state is large (e.g., image pixels).
        args:
        X0: initial state vector
        Q_var: process noise variance (scalar)
        R_var: measurement noise variance (scalar)
        
        details:
        The Kalman gain is computed once thanks to the Riccati solution of the Kalman equations.
        Can be shown that the covariance P converge to this solution

        P = (Q + np.sqrt(Q**2 + 4*Q*self.R))/2

        and so the Kalman Gain converges to its tipical function of P:
        K = P / (P + R)
        
        '''
        self.X = X0
        self.Q = Q_var
        self.R = R_var
        self.P = (self.Q + np.sqrt(self.Q**2 + 4*self.Q*self.R))/2
        self.K = self.P / (self.P + self.R)

    def update(self, Z):
        self.X = self.X + self.K * (Z - self.X)
        return self.X.flatten()



class big_simple_adaptive_kalman:
    '''
    A simple Kalman filter for large state vectors.
    It assumes a simple model with no covariance between the state variables,
    where all the state variables are observed, with equal measurement variance.
    The process variance is computed as the innovation squared.
    In this way, the Kalman gain can be computed as a vector, 
    which makes the computation feasible for hundreds of thousands of variables.
    This is suitable for denoising applications where the state is large (e.g., image pixels).
    This filter is more robust to blurry images than the filter of big_simple_kalman.
    X0: initial state vector
    P0: initial estimate covariance (scalar). Is assumed initially equal for all variables, 
    and then is updated with specific Q values (Q=Y^2 where Y is the innovation-error)
    R: measurement noise variance (scalar)'''
    def __init__(self, X0, P0, R):
        self.X = X0.flatten()
        self.P = np.ones(X0.shape[0])*P0
        self.R = R

    def update(self, Z):
        Y = Z - self.X
        Q = Y**2
        self.P = self.P + Q
        K = self.P / (self.P + self.R)
        self.X = self.X + K*Y
        self.P = (1-K)*self.P
        return self.X






class BigAdaptiveKalman:

    def __init__(self, X0,
                 P0=1.0,
                 Q_base=0.5,
                 Q_motion=10.0,
                 R=10.0,
                 motion_threshold=20.0):

        self.X = X0.astype(np.float32)

        # Per-pixel variance
        self.P = np.ones_like(self.X, dtype=np.float32) * P0

        self.Q_base = Q_base
        self.Q_motion = Q_motion
        self.R = R
        self.motion_threshold = motion_threshold

        # buffer to reuse innovation
        self.innovation = np.zeros_like(self.X, dtype=np.float32)


    def _compute_adaptive_Q(self):
        """
        Compute Q map based on innovation magnitude.
        """
        motion_mask = np.abs(self.innovation) > self.motion_threshold

        Q_adaptive = np.where(motion_mask,
                              self.Q_motion,
                              self.Q_base)

        return Q_adaptive


    def predict(self):
        """
        Prediction step with adaptive Q.
        Assumes innovation has already been computed.
        """

        Q_adaptive = self._compute_adaptive_Q()

        self.P += Q_adaptive


    def update(self, Z):
        """
        Full Kalman step (innovation + predict + update)
        """

        # 1️⃣ Compute innovation once
        self.innovation = Z - self.X

        # 2️⃣ Predict (uses innovation to adapt Q)
        self.predict()

        # 3️⃣ Compute gain
        K = self.P / (self.P + self.R)

        # 4️⃣ Update state
        self.X += K * self.innovation

        # 5️⃣ Update covariance
        self.P *= (1 - K)

        return self.X






class KalmanAdaptiveKernel2d:
    def __init__(self, X0, P0, R, kernel=None, kc=1, ks=1, kd=1, Yconv=False, Qconv=True):

        '''
        X0: 2d numpy matrix
        P0: scalar
        R: scalar
        kc, ks, kd: scalars used to build the default 3x3 kernel
        kernel: small 2d numpy matrix, a kernel filter
        '''

        self.X = X0
        self.P = np.ones(X0.shape)*P0
        self.R = R
        self.Yconv = Yconv
        self.Qconv = Qconv
        if kernel is None:
            self.kernel = np.array([[kd,ks,kd],
                                    [ks,kc,ks],
                                    [kd,ks,kd]])
        else:
            self.kernel = kernel
        self.kernel = self.kernel/self.kernel.flatten().sum()


    def update(self, Z):
        '''
        Z: numpy matrix, that has to be denoised over time
        '''

        Y = Z - self.X
        Ys = Y**2
        if self.Qconv:
            Q = convolve2d(Ys, self.kernel, mode='same', boundary='symm')
        else:
            Q = Ys
        self.P += Q

        if self.Yconv:
            Y = convolve2d(Y, self.kernel, mode='same', boundary='symm')

        K = self.P / (self.P + self.R)
        self.X += K*Y

        self.P = (1-K)*self.P

        return self.X













class big_simple_kalman_2d:
    def __init__(self, X0, P0=1, Q_var=3, R_var=10, kernel=None, kc=1, ks=1, kd=1):

        '''
        A simple Kalman filter for large state matrices. 
        It is similar to big_simple_kalman, but adopts a blur method on the error matrix.
        Here X0, X and Z are 2d matrices.
        It assumes a simple model with no covariance between the state variables,
        where all the state variables are observed, with equal measurement and process variance. 
        In this way, the Kalman gain is the can be computed as a scalar, 
        which makes the computation feasible for hundreds of thousands of variables.
        This is suitable for denoising applications where the state is large (e.g., image pixels).
        X0: initial state vector
        P0: initial estimate covariance (scalar)
        Q_var: process noise variance (scalar)
        R_var: measurement noise variance (scalar)
        kernel: default None, (optional) 2d matrix (numpy) to use to filter the innovation matrix,
        if is None, then a 3x3 kernel is build using kc (at the center), ks (for the 4 sides), kd (for the 4 angles)
        kc: int default 1, to build the default kernel
        ks: int default 1, to build the default kernel
        kd: int default 1, to build the default kernel
        '''
    
        self.X = X0
        self.P = P0
        self.Q = Q_var
        self.R = R_var
        if kernel is None:
            self.kernel = np.array([[kd,ks,kd],
                                    [ks,kc,ks],
                                    [kd,ks,kd]])
        else:
            self.kernel = kernel
        self.kernel = self.kernel/self.kernel.flatten().sum()

        

    def update(self, Z):
        '''
        Z: numpy matrix, that has to be denoised over time
        '''
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        Y = Z - self.X
        Y = convolve2d(Y, self.kernel, mode='same', boundary='symm')
        self.X = self.X + K * Y
        self.P = (1 - K) * self.P
        return self.X










class KalmanKernel2d:
    def __init__(self, X0, Var0,Cov0, QVar, QCov, R):
        '''
        Kalman filter to denoise a sequence of matrices (e.g. a video, grayscale or rgb single channel)
        This function tries to build an optimal kernel to adjust each matrix by a noise.
        
        X0: 2d matrix (np array), initialization of the state X
        Var: initial variance of a cell of the matrix
        Cov: initial covariance of two adjacent cells
        QVar: state noise (strength of measurement). Increase variance of cell over time.
        QCov: similar to QVar, increase the covariance over time.
        R: measurement noise (strength of prior)

        Based on the assumptions of the Kalman Filter, this model assumes that both 
        F (transition matrix) and H (matrix that selects the observed components of the state)
        are indentities. In this way we assume that all the cells are observed, with no latent variables (H)
        and that there is no deterministic way that makes the state X to evolve over time (F)

        see help for update for mathematical details
        '''
        self.X = X0
        self.V = Var0
        self.C = Cov0
        self.R = R
        self.QV = QVar
        self.QC = QCov
        

    def update(self, Z):
        '''
        Z is a 2d matrix (numpy), with the same shape of X0 (from init) that evolves over time
        This function tries to denoise it based on the old denoised values of it,
        and returns the real state X

        the formula for the kernel reflects the update equation for the kalman gain
        K = P(P+R)^(-1)

        but is described in this way:

        P = [
        [C,C,C],
        [C,V,C],
        [C,C,C]
        ]

        K = P / (R + V + 8C)

        Where V is the variance of each cell, and C is the covariance between two adjacent cells
        (in this way I can interpret P as the covariance matrix of the Kalman Filter, but really
        this formulation is nearer Wiener Filter).
        We should update P in a way similar to the prediction step of the Kalman Filter:
        
        P = FPF+Q

        Here we assume F to be the identity matrix (so we just want to denoise X, not to predict it).
        then P = P + Q. But here Q is a scalar.

        Update P is equivalent to updating the terms C and V.
        

        defining 

        K_v = V / (R + V + 8C)
        K_c = C / (R + V + 8C)

        
        then the update equations for the covariance matrix P become

        V = V*(1-K_v)+Q
        C = C*(1-K_c)
        '''

        ##compute the kalman gain / kernel filter
        PRinv = self.R + self.V + self.C*8 
        K_v = self.V / PRinv
        K_c = self.C / PRinv
        K = np.ones((3,3))*K_c
        K[1,1] = K_v

        #update the state matrix
        Y = Z - self.X
        self.X = self.X + convolve2d(Y, K, mode='same', boundary='symm')

        #update variance and covariance
        self.V = (1-K_v)*self.V + self.QV
        self.C = (1-K_c)*self.C + self.QC

        return self.X






class common_simple_kalman_filter():
    def __init__(self , X0, P0, R, Q):
        '''
        Assumes that X is a list of state-space vectors with the same covariance matrix.
        Is simple because F and H are assumed to be identities.

        X0: 2d matrix, with the state variables on the second axis
        and all the objects (with common parameters) on the first axis
        P0: square numpy matrix, initial covariance
        R: vector of measurement noise (length of state)
        Q: vector of process noise (length of state)
        '''
        
        self.X = X0
        self.P = P0
        self.dims = X0.shape[1]
        self.R = R
        self.Q = Q

    def update(self, Z):
        '''
        Z: numpy matrix with the same shape of X0
        '''

        Y = Z - self.X
        self.P = self.P + self.Q
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + Y @ K
        self.P = (np.eye(self.dims)-K) @ self.P

        return self.X




class common_adaptive_kalman_filter():

    def __init__(self , X0, P0, R, q_base=1.0, alpha=0.05):

        self.X = X0
        self.P = P0
        self.dims = X0.shape[1]
        self.R = R

        self.Q0 = q_base * np.eye(self.dims)
        self.lambda_k = 1.0
        self.alpha = alpha

    def update(self, Z):

        Y = Z - self.X

        # --- compute NIS ---
        S = self.P + self.R
        Sinv = np.linalg.inv(S)

        nis = np.mean(np.sum((Y @ Sinv) * Y, axis=1)) / self.dims

        # --- adaptive scaling ---
        lambda_target = np.clip(nis, 0.5, 10.0)
        self.lambda_k = (1-self.alpha)*self.lambda_k + self.alpha*lambda_target

        self.Q = self.lambda_k * self.Q0

        # --- standard KF ---
        self.P = self.P + self.Q
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.X = self.X + Y @ K
        self.P = (np.eye(self.dims)-K) @ self.P

        return self.X



class common_adaptive_kalman_filter0():
    def __init__(self , X0, P0, R, alpha=0.01):
        '''
        Assumes that X is a list of state-space vectors with the same covariance matrix.
        Is simple because F and H are assumed to be identities.
        Q is adaptive, estimated from innovation.

        X0: 2d matrix, with the state variables on the second axis
        and all the objects (with common parameters) on the first axis
        P0: square numpy matrix, initial covariance
        R: vector of measurement noise (length of state)
        '''
        
        self.X = X0
        self.P = P0
        self.dims = X0.shape[1]
        self.R = R
        self.Q = np.eye(self.dims) + 1
        self.alpha=alpha

    def update(self, Z):
        '''
        Z: numpy matrix with the same shape of X0
        '''
        Y = Z - self.X
        
        Ycov = Y.T @ Y  / Y.shape[0]
        self.Q = (1-self.alpha)*self.Q + self.alpha*Ycov

        print(self.Q)

        self.P = self.P + self.Q
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.X = self.X + Y @ K
        self.P = (np.eye(self.dims)-K) @ self.P

        return self.X







from kalman_filter_class import big_simple_kalman, big_simple_adaptive_kalman, big_simple_steady_kalman, common_simple_kalman_filter


def kalman_matrix_denoiser41(mat, R=400, Q=100, mode='both', type='steady', once=False):

    '''
    mat: matrix 2d
    R: scalar
    Q: scalar
    mode: str, default 'both', can be ('both', 'hor', 'ver', 'once').
    If both then 4 kalman filters are produced, 2 sliding vertically over the image and two horizontally.
    If ver or hor only in one two kalman filters are used to denoise the image in both directions on a particular axis
    type: str, default 'steady', the type of Kalman filter to use for each side and channel.
    Can be ('steady', 'adaptive', 'simple')
    once: bool, default False. If True, only one filter per axis is computed. 
    '''

    image0 = np.array(mat).astype(np.float32)
    N0 = image0.shape[0]
    N1 = image0.shape[1]

    assert mode in ['both', 'ver', 'hor'], 'mode can be (both, hor, ver) '
    assert type in ['steady', 'adaptive', 'simple'], 'the KF type can be (steady, adaptive, simple)'

    if mode=='both':
        vertical=True
        horizontal=True
    elif mode=='ver':
        vertical=True
        horizontal=False
    elif mode=='hor':
        vertical=False
        horizontal=True     

    
    steady = (type=='steady')
    adaptive = (type=='adaptive')

    if vertical:
        if steady:
            kf1 = big_simple_steady_kalman(X0 = image0[1,:], Q_var=Q, R_var=R)
            kf3 = big_simple_steady_kalman(X0 = image0[N0-2,:], Q_var=Q, R_var=R)
        elif adaptive:
            kf1 = big_simple_adaptive_kalman(X0 = image0[1,:], P0=1, R=R)
            kf3 = big_simple_adaptive_kalman(X0 = image0[N0-2,:], P0=1, R=R)
        else:
            kf1 = big_simple_kalman(X0 = image0[1,:], Q_var=Q, R_var=R)
            kf3 = big_simple_kalman(X0 = image0[N0-2,:], Q_var=Q, R_var=R)

        image1w = image0.copy()
        image2w = image0.copy()

        if once:
            for i in range(N0):
                new = image0[i,:]
                new0 = kf1.update(new)
                image1w[i,:] = new0        
        else:
            for i in range(N0):
                new = image0[i,:]
                new0 = kf1.update(new)
                image1w[i,:] = new0

                v = N0 - i - 1
                new = image0[v,:]
                new0 = kf3.update(new)
                image2w[v,:] = new0


    if horizontal:
        if steady:
            kf2 = big_simple_steady_kalman(X0 = image0[:,1], Q_var=Q, R_var=R)
            kf4 = big_simple_steady_kalman(X0 = image0[:,N1-2], Q_var=Q, R_var=R)
        elif adaptive:
            kf2 = big_simple_adaptive_kalman(X0 = image0[:,1], P0=1, R=R)
            kf4 = big_simple_adaptive_kalman(X0 = image0[:,N1-2], P0=1, R=R)
        else:
            kf2 = big_simple_kalman(X0 = image0[:,1], Q_var=Q, R_var=R)
            kf4 = big_simple_kalman(X0 = image0[:,N1-2], Q_var=Q, R_var=R)

        image1h = image0.copy()
        image2h = image0.copy()

        if once:
            for i in range(N1):
                new = image0[:,i]
                new0 = kf2.update(new)
                image1h[:,i] = new0
        else:
            for i in range(N1):
                new = image0[:,i]
                new0 = kf2.update(new)
                image1h[:,i] = new0

                v = N1 - i - 1
                new = image0[:,v]
                new0 = kf4.update(new)
                image2h[:,v] = new0


    if once:
        if mode=='both':
            image2 = (image1w + image1h) / 2
        elif mode=='ver':
            image2 = image1w
        elif mode=='hor':
            image2 = image1h
    else:
        if mode=='both':
            image2 = (image1w + image1h + image2w + image2h) / 4
        elif mode=='ver':
            image2 = (image1w + image2w) / 2
        elif mode=='hor':
            image2 = (image1h + image2h) / 2

    return image2




class kalman_video_denoiser():
    def __init__(self, shape, R=400, X0=None, P0=None):

        dim = 1
        for d in shape:
            dim *= d

        self.dim = dim
        self.shape = shape

        if X0 is None:
            X0 = np.zeros(self.dim)

        if P0 is None:
            P0 = 1

        self.kf = big_simple_adaptive_kalman(X0=X0, P0=P0, R=R)


    def update(self, frame):
        frame_flat = np.array(frame).flatten()
        frame_flat = self.kf.update(frame_flat)
        frame0 = frame_flat.reshape(self.shape).astype(np.uint8)
        return frame0




def kalman_rgb_img_denoiser41(image, R=400, Q=100, mode='both', type='steady', once=True):
    image = image.copy()
    image0 = image[:,:,0]
    image1 = image[:,:,1]
    image2 = image[:,:,2]

    image0d = kalman_matrix_denoiser41(image0, R=R, Q=Q, mode=mode, type=type, once=once).astype(np.uint8)
    image1d = kalman_matrix_denoiser41(image1, R=R, Q=Q, mode=mode, type=type, once=once).astype(np.uint8)
    image2d = kalman_matrix_denoiser41(image2, R=R, Q=Q, mode=mode, type=type, once=once).astype(np.uint8)

    image[:,:,0] = image0d
    image[:,:,1] = image1d
    image[:,:,2] = image2d
    return image


def kalman_rgb_img_denoiser42(image, P0=1, R_var=400, Q_var=120, Q_cov=80, mode='both', once=False):
    '''
    image: an RGB image to denoise, of shape (N0,N1,3)
    R_var: scalar
    Q_var: scalar
    Q_cov: scalar
    mode: str, default 'both', can be ('both', 'hor', 'ver', 'once').
    If both then 4 kalman filters are produced, 2 sliding vertically over the image and two horizontally.
    If ver or hor only in one two kalman filters are used to denoise the image in both directions on a particular axis
    once: bool, default False. If True, only one filter per axis is computed. 
    '''


    image = np.array(image.copy()).astype(np.float32)

    N0 = image.shape[0]
    N1 = image.shape[1]

    assert mode in ['both', 'ver', 'hor'], 'mode can be (both, hor, ver) '

    if mode=='both':
        vertical=True
        horizontal=True
    elif mode=='ver':
        vertical=True
        horizontal=False
    elif mode=='hor':
        vertical=False
        horizontal=True     


    P0 = np.eye(3)*P0
    R = np.eye(3)*R_var
    Q = np.eye(3)*Q_var + Q_cov - np.eye(3)*Q_cov


    if vertical:

        kf1 = common_simple_kalman_filter(X0=image[1,:,:],P0=P0,R=R,Q=Q)
        kf2 = common_simple_kalman_filter(X0=image[N0-1,:,:],P0=P0,R=R,Q=Q)

        image1w = image.copy()
        image2w = image.copy()

        if once:
            for i in range(N0):
                image1w[i,:,:] = kf1.update(image[i,:,:])            
        else:
            for i in range(N0):
                image1w[i,:,:] = kf1.update(image[i,:,:])

                v = N0 - i - 1
                image2w[v,:,:] = kf2.update(image[v,:,:])


    if horizontal:
        kf3 = common_simple_kalman_filter(X0=image[:,1,:],P0=P0,R=R,Q=Q)
        kf4 = common_simple_kalman_filter(X0=image[:,N1-1,:],P0=P0,R=R,Q=Q)

        image1h = image.copy()
        image2h = image.copy()

        if once:
            for i in range(N1):
                image1h[:,i,:] = kf3.update(image[:,i,:])            
        else:
            for i in range(N1):
                image1h[:,i,:] = kf3.update(image[:,i,:])

                v = N1 - i - 1
                image2h[:,v,:] = kf4.update(image[:,v,:])


    if once:
        if mode=='both':
            image2 = (image1w + image1h) / 2
        elif mode=='ver':
            image2 = image1w
        elif mode=='hor':
            image2 = image1h
    else:
        if mode=='both':
            image2 = (image1w + image1h + image2w + image2h) / 4
        elif mode=='ver':
            image2 = (image1w + image2w) / 2
        elif mode=='hor':
            image2 = (image1h + image2h) / 2

    return image2.astype(np.uint8)














    def get_block_indices(W, H, w, h):
        # 1. Create a grid of coordinates (i, j) for the whole matrix
        i, j = np.indices((W, H))
        
        # 2. Calculate the size of each individual block in pixels
        block_height = W // w
        block_width = H // h
        
        # 3. Find the block coordinates for each pixel
        block_row = i // block_height
        block_col = j // block_width
        
        # 4. Map (block_row, block_col) to a single index from 0 to N-1
        # Standard row-major indexing: index = row * total_columns + col
        block_idx_matrix = block_row * h + block_col
        
        # 5. Flatten to match your vector
        return block_idx_matrix.flatten()


class SORT:
    def __init__(self, X0, P0, F, Q, H, R, max_missed=3, min_hit=3, metric='euclidean',
                 save_measurement=False, save_state=False):
        """
        SORT algorithm (Simple Online Realtime Tracker)
        Multiple object tracker using Kalman Filter with adjustable parameters
        At each timestep, given a set of measurements and a set of trackers (kalman filters),
        the SORT will assign each measurement to each tracker, following the hungarian method.
        If there are unassigned measurements, for each of them a new tracker will be generated.
        If there are unassigned trackers, the missed count will increase by 1. 
        If it will be greater than max_missed, then the track will be deleted.
        
        X0: (Kalman Filter argument) state vector at time 0 of dim n : will evolve over time
        P0: (Kalman Filter argument) initial covariance matrix of dim n x n : will evolve over time
        F: (Kalman Filter argument) state transition matrix of dim n x n
        H: (Kalman Filter argument) measurement matrix of dim m x n
        Q: (Kalman Filter argument) process noise covariance matrix of dim n x n
        R: (Kalman Filter argument) measurement noise covariance matrix of dim m x m
        Z: (Kalman Filter argument) measurement vector (dimension less or equal to state vector) of dim m
        max_missed: int, maximum number of consecutive misses for
        a single tracker before it will be 
        min_hint: int, minimum number of measurement assignments to a track, to make it visible
        metric: str, argument for scipy.spatial.distance.cdist. Default 'euclidean'.
        """
        self.X0 = X0
        self.P0 = P0
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.max_missed = max_missed
        self.min_hit = min_hit
        self.n_measurements = 0
        self.n_tracks_now = 0
        self.metric = metric
        self.tracks_history = {}
        self.time_id = 0

        self.save_measurement = save_measurement
        self.save_state = save_state
        self.trackers = {}
        self.n_tracks_total = 0
        self.ndim = X0.shape[0]


    def create_track(self, x0):
        track = KalmanFilter(x0, self.P0, self.F, self.H, self.Q, self.R)
        track.hit = 0
        track.missed = 0
        track.isactive = False
        track.track_id = self.n_tracks_total
        self.tracks_history[track.track_id] = {'measurement':{}, 'state':{}, 'state_arr': np.zeros((self.ndim, 0), dtype=float), 'meta':{}}
        self.n_tracks_total +=1
        self.trackers[track.track_id] = track
        

    def update_track(self, track, measurement, metadata=None):
        track.predict()
        track.update(measurement)
        track.hit += 1
        track.missed = 0
        if not track.isactive:
            if track.hit > self.min_hit:
                track.isactive = True

        #self.tracks_history[track.track_id][self.time_id] = {}

        if self.save_measurement:
            self.tracks_history[track.track_id]['measurement'][self.time_id] = measurement

        if self.save_state:
            self.tracks_history[track.track_id]['state'][self.time_id] = track.X.copy()
            self.tracks_history[track.track_id]['state_arr'] = np.hstack(
                (self.tracks_history[track.track_id]['state_arr'], track.X.copy()))

        if metadata is not None:
            self.tracks_history[track.track_id]['meta'][self.time_id] = metadata

    def get_trackers(self):
        '''function to get the observed components of the kalman filter of all the trackers'''
        ids = list(self.trackers.keys())
        data = {id: self.trackers[id].get_state_observed() for id in ids}
        return data

    def get_trackers_array(self):
        return np.array([t.get_state_observed() for t in list(self.trackers.values())])

    def get_active_trackers(self):
        '''function to get the observed components of the kalman filter of the stable trackers'''
        ids = [id for id in list(self.trackers.keys()) if self.trackers[id].isactive]
        data = {id: self.trackers[id].get_state_observed() for id in ids}
        return data
        #return np.array([t.get_state_observed() for t in self.trackers if t.isactive])

    def get_active_trackers_history(self):
        ids = list(self.trackers.keys())#[t.track_id for t in list(self.trackers.values()) if t.isactive]
        data = {id: self.tracks_history[id] for id in ids}
        return data

    def update_tracks(self, measurments, metadata_list=None):
        '''
        measurments: List of measurement vectors (each of shape (m,))
        metadata_list: List of metadata associated with each measurement (optional)
        '''

        nme = len(measurments)
        ntr = len(self.trackers)
        self.n_measurements = nme
        self.n_tracks_now = ntr

        if metadata_list is None:
            metadata_list = [None] * nme

        self.time_id += 1
        measurments = np.array(measurments).reshape((nme, self.ndim))
        

        if nme>0:
            if ntr>0:
                track_vals = self.get_trackers_array()
                track_vals = track_vals.reshape((ntr, self.ndim))
                distmat = cdist(track_vals, measurments, metric='euclidean')
                track_id, me_id = linear_sum_assignment(distmat)

                real_ids = list(self.trackers.keys())
                track_id0 = [real_ids[i] for i in track_id]
                track_id = track_id0

                n_tracks = track_vals.shape[0]
                n_me = measurments.shape[0]

                all_tracks = real_ids #np.arange(n_tracks)
                all_me = np.arange(n_me)

                ##record missed assignments
                missed_tracks = np.setdiff1d(all_tracks, track_id)
                missed_me = np.setdiff1d(all_me, me_id)

                for i in range(len(track_id)):
                    self.update_track(self.trackers[int(track_id[i])], measurments[me_id[i]], metadata=metadata_list[me_id[i]])

                if (len(missed_tracks)>len(missed_me)):
                    id_to_remove = []
                    for i in range(len(missed_tracks)):
                        track_id01 = int(missed_tracks[i])
                        tr = self.trackers[track_id01]
                        tr.missed += 1
                        if (tr.missed > self.max_missed):
                            id_to_remove.append(track_id01)
                
                    for idx in sorted(id_to_remove, reverse=True):
                        self.trackers.pop(idx)

                else:
                    for id_miss in missed_me:
                        self.create_track(measurments[id_miss])
            else:
                for m in measurments:
                    self.create_track(m)
        else:
            if len(self.trackers)>0:
                id_to_remove = []
                for i, k in enumerate(self.trackers):
                    t = self.trackers[k]
                    t.predict()
                    t.missed += 1
                    if (t.missed > self.max_missed):
                        id_to_remove.append(k)
                
                for idx in sorted(id_to_remove, reverse=True):
                    self.trackers.pop(idx)








class SORT000:
    def __init__(self, X0, P0, F, Q, H, R, max_missed=3, min_hit=3, metric='euclidean',
                 save_measurement=False, save_state=False):
        """
        SORT algorithm (Simple Online Realtime Tracker)
        Multiple object tracker using Kalman Filter with adjustable parameters
        At each timestep, given a set of measurements and a set of trackers (kalman filters),
        the SORT will assign each measurement to each tracker, following the hungarian method.
        If there are unassigned measurements, for each of them a new tracker will be generated.
        If there are unassigned trackers, the missed count will increase by 1. 
        If it will be greater than max_missed, then the track will be deleted.
        
        X0: (Kalman Filter argument) state vector at time 0 of dim n : will evolve over time
        P0: (Kalman Filter argument) initial covariance matrix of dim n x n : will evolve over time
        F: (Kalman Filter argument) state transition matrix of dim n x n
        H: (Kalman Filter argument) measurement matrix of dim m x n
        Q: (Kalman Filter argument) process noise covariance matrix of dim n x n
        R: (Kalman Filter argument) measurement noise covariance matrix of dim m x m
        Z: (Kalman Filter argument) measurement vector (dimension less or equal to state vector) of dim m
        max_missed: int, maximum number of consecutive misses for
        a single tracker before it will be 
        min_hint: int, minimum number of measurement assignments to a track, to make it visible
        metric: str, argument for scipy.spatial.distance.cdist. Default 'euclidean'.
        """
        self.X0 = X0
        self.P0 = P0
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.max_missed = max_missed
        self.min_hit = min_hit
        self.n_measurements = 0
        self.n_tracks_now = 0
        self.metric = metric
        self.tracks_history = {}
        self.time_id = 0

        self.save_measurement = save_measurement
        self.save_state = save_state
        self.trackers = []
        self.n_tracks_total = 0
        self.ndim = X0.shape[0]


    def create_track(self, x0):
        track = KalmanFilter(x0, self.P0, self.F, self.H, self.Q, self.R)
        track.hit = 0
        track.missed = 0
        track.isactive = False
        track.track_id = self.n_tracks_total
        self.tracks_history[track.track_id] = {'measurement':{}, 'state':{}, 'state_arr': np.zeros((self.ndim, 0), dtype=float), 'meta':{}}
        self.n_tracks_total +=1
        self.trackers.append(track)
        

    def update_track(self, track, measurement, metadata=None):
        track.predict()
        track.update(measurement)
        track.hit += 1
        track.missed = 0
        if not track.isactive:
            if track.hit > self.min_hit:
                track.isactive = True

        self.tracks_history[track.track_id][self.time_id] = {}

        if self.save_measurement:
            self.tracks_history[track.track_id]['measurement'][self.time_id] = measurement

        if self.save_state:
            # print('track shape')
            # print(track.X.shape)
            # print('db shape')
            # print(self.tracks_history[track.track_id]['state_arr'].shape)
            self.tracks_history[track.track_id]['state'][self.time_id] = track.X.copy()
            self.tracks_history[track.track_id]['state_arr'] = np.hstack(
                (self.tracks_history[track.track_id]['state_arr'], track.X.copy()))

        if metadata is not None:
            self.tracks_history[track.track_id]['meta'][self.time_id] = metadata

    def get_trackers(self):
        '''function to get the observed components of the kalman filter of all the trackers'''
        return np.array([t.get_state_observed() for t in self.trackers])

    def get_active_trackers(self):
        '''function to get the observed components of the kalman filter of the stable trackers'''
        return np.array([t.get_state_observed() for t in self.trackers if t.isactive])

    def get_active_trackers_history(self):
        ids = [t.track_id for t in self.trackers if t.isactive]
        data = {id: self.tracks_history[id] for id in ids}
        return data

    def update_tracks(self, measurments, metadata_list=None):
        '''
        measurments: List of measurement vectors (each of shape (m,))
        metadata_list: List of metadata associated with each measurement (optional)
        '''

        nme = len(measurments)
        ntr = len(self.trackers)
        self.n_measurements = nme
        self.n_tracks_now = ntr

        if metadata_list is None:
            metadata_list = [None] * nme

        self.time_id += 1
        measurments = np.array(measurments).reshape((nme, self.ndim))
        

        if nme>0:
            if ntr>0:
                track_vals = self.get_trackers()
                track_vals = track_vals.reshape((ntr, self.ndim))
                distmat = cdist(track_vals, measurments, metric='euclidean')
                track_id, me_id = linear_sum_assignment(distmat)

                n_tracks = track_vals.shape[0]
                n_me = measurments.shape[0]

                all_tracks = np.arange(n_tracks)
                all_me = np.arange(n_me)

                ##record missed assignments
                missed_tracks = np.setdiff1d(all_tracks, track_id)
                missed_me = np.setdiff1d(all_me, me_id)

                for i in range(len(track_id)):
                    self.update_track(self.trackers[track_id[i]], measurments[me_id[i]], metadata=metadata_list[me_id[i]])

                if (len(missed_tracks)>len(missed_me)):
                    id_to_remove = []
                    for i in range(len(missed_tracks)):
                        track_id = missed_tracks[i]
                        tr = self.trackers[track_id]
                        tr.missed += 1
                        if (tr.missed > self.max_missed):
                            id_to_remove.append(track_id)
                
                    for idx in sorted(id_to_remove, reverse=True):
                        self.trackers.pop(idx)

                else:
                    for id_miss in missed_me:
                        self.create_track(measurments[id_miss])
            else:
                for m in measurments:
                    self.create_track(m)
        else:
            if len(self.trackers)>0:
                id_to_remove = []
                for i in range(len(self.trackers)):
                    t = self.trackers[i]
                    t.predict()
                    t.missed += 1
                    if (t.missed > self.max_missed):
                        id_to_remove.append(i)
                
                for idx in sorted(id_to_remove, reverse=True):
                    self.trackers.pop(idx)
