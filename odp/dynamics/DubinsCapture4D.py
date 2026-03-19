import heterocl as hcl


class DubinsCapture4D:
    """
        Pursuit-evasion game dynamics for Dubins cars (4D)
        
        Evader: Dubins4D (has velocity as state variable)
        Pursuer: Dubins3D (constant velocity)
        
        x_dot = -v + speed*cos(theta) + w*y
        y_dot = speed*sin(theta) - w*x
        theta_dot = d - w
        v_dot = a
        
        where:
        - state = [x, y, theta, v]: relative position, relative heading, evader velocity
        - control = [w, a]: evader's angular velocity and acceleration
        - disturbance = [d]: pursuer's angular velocity
        - speed: pursuer's constant velocity
    """
    def __init__(self, x=None, wMax=2.0, aMax=2.0, aMin=-2.0, vMax=1.0, vMin=0.1, 
                 speed=1.0, dMax=2.0, uMode="max", dMode="min"):
        if x is None:
            x = [0, 0, 0, 1.0]  # [x, y, theta, v]
        self.x = x
        self.wMax = wMax
        self.aMax = aMax
        self.aMin = aMin
        self.vMax = vMax
        self.vMin = vMin
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
                :param  spat_deriv: tuple of spatial derivative in all dimensions
                        state: [x, y, theta, v]
                        t: time
                :return: a tuple of optimal controls [w, a]
        """

        opt_w = hcl.scalar(self.wMax, "opt_w")
        opt_a = hcl.scalar(self.aMax, "opt_a")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        # Compute optimal angular velocity term
        # a_term = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]
        a_term = hcl.scalar(0, "a_term")
        a_term[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(a_term >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w[0]
        with hcl.elif_(a_term < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w[0]

        # Compute optimal acceleration term
        # b_term = spat_deriv[3] (derivative w.r.t. velocity)
        b_term = hcl.scalar(0, "b_term")
        b_term[0] = spat_deriv[3]

        with hcl.if_(b_term >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_a[0] = self.aMin
        with hcl.elif_(b_term < 0):
            with hcl.if_(self.uMode == "max"):
                opt_a[0] = self.aMin
            with hcl.else_():
                opt_a[0] = self.aMax

        return (opt_w[0], opt_a[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
            :param spat_deriv: tuple of spatial derivative in all dimensions
                    state: [x, y, theta, v]
                    t: time
            :return: a tuple of optimal disturbances [d]
        """

        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(self.dMax, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        # Compute optimal pursuer angular velocity
        # c_term = spat_deriv[2] (derivative w.r.t. theta)
        c_term = hcl.scalar(0, "c_term")
        c_term[0] = spat_deriv[2]

        with hcl.if_(c_term[0] >= 0):
            with hcl.if_(self.dMode == "min"):
                d1[0] = -d1[0]
        with hcl.elif_(c_term[0] < 0):
            with hcl.if_(self.dMode == "max"):
                d1[0] = -d1[0]
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        v_dot = hcl.scalar(0, "v_dot")

        # x_dot = -v + speed*cos(theta) + w*y
        x_dot[0] = -state[3] + self.speed*hcl.cos(state[2]) + uOpt[0]*state[1]
        # y_dot = speed*sin(theta) - w*x
        y_dot[0] = self.speed*hcl.sin(state[2]) - uOpt[0]*state[0]
        # theta_dot = d - w
        theta_dot[0] = dOpt[0] - uOpt[0]
        # v_dot = a
        v_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], theta_dot[0], v_dot[0])

    # The below function can have whatever form or parameters users want
    # These functions are not used in HeteroCL program, hence is pure Python code and
    # can be used after the value function has been obtained.
    def optCtrl_inPython(self, state, spat_deriv):
        a_term = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]
        b_term = spat_deriv[3]

        opt_w = self.wMax
        if a_term >= 0:
            if self.uMode == "min":
                opt_w = -self.wMax
        else:
            if self.uMode == "max":
                opt_w = -self.wMax

        opt_a = self.aMax
        if b_term >= 0:
            if self.uMode == "min":
                opt_a = self.aMin
        else:
            if self.uMode == "max":
                opt_a = self.aMin
            else:
                opt_a = self.aMax

        return (opt_w, opt_a)