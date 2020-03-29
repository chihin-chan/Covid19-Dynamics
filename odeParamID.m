function bestc = odeParamID()
%% Create some "experimental data"
expTime = 0:0.01:10;
expY = exp(-expTime) + 0.02*randn(size(expTime));
plot(expTime, expY)

%% ODE Information
tSpan = [0 10];
z0 = 1; % Initial values for the state

%% Initial guess
c0 = 0.5; % First guess at parameter value
ODE_Sol = ode45(@(t,z)updateStates(t,z,c0), tSpan, z0); % Run the ODE
simY = deval(ODE_Sol, expTime); % Evaluate the solution at the experimental time steps

hold on
plot(expTime, simY, '-r')

%% Set up optimization
myObjective = @(x) objFcn(x, expTime, expY,tSpan,z0);
lb = 0;
ub = 5;

bestc = lsqnonlin(myObjective, c0, lb, ub);

%% Plot best result
ODE_Sol = ode45(@(t,z)updateStates(t,z,bestc), tSpan, z0);
bestY = deval(ODE_Sol, expTime);

plot(expTime, bestY, '-g')
legend('Exp Data','Initial Param','Best Param');

function f = updateStates(t, z, c)

f = -c*z;

function cost = objFcn(x,expTime,expY,tSpan,z0)

ODE_Sol = ode45(@(t,z)updateStates(t,z,x), tSpan, z0);
simY = deval(ODE_Sol, expTime);

cost = simY-expY;

