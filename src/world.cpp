#include "world.h"

world::world(string name, int n) {

    setInput m_inputData;
    m_inputData = setInput();
    m_inputData.LoadOptions(name);

    // render = m_inputData.GetBoolOpt("render");                       // boolean
    RodLength = m_inputData.GetScalarOpt("RodLength");               // meter
    // helixradius = m_inputData.GetScalarOpt("helixradius");           // meter
    gVector = m_inputData.GetVecOpt("gVector");                            // m/s^2
    maxIter = m_inputData.GetIntOpt("maxIter");                      // maximum number of iterations
    // helixpitch = m_inputData.GetScalarOpt("helixpitch");             // meter
    rodRadius = m_inputData.GetScalarOpt("rodRadius");               // meter
    // numVertices = m_inputData.GetIntOpt("numVertices");              // int_num
    numVertices = n;
    youngM = m_inputData.GetScalarOpt("youngM");                     // Pa
    Poisson = m_inputData.GetScalarOpt("Poisson");                   // dimensionless
    deltaTime = m_inputData.GetScalarOpt("deltaTime");               // seconds
    tol = m_inputData.GetScalarOpt("tol");                           // small number like 10e-7
    stol = m_inputData.GetScalarOpt("stol");                         // small number, e.g. 0.1%
    density = m_inputData.GetScalarOpt("density");                   // kg/m^3
    viscosity = m_inputData.GetScalarOpt("viscosity");               // viscosity in Pa-s
    // data_resolution = m_inputData.GetScalarOpt("dataResolution");    // time resolution for recording data
    // pull_time = m_inputData.GetScalarOpt("pullTime");                // get time of pulling
    // release_time = m_inputData.GetScalarOpt("releaseTime");          // get time of loosening
    // wait_time = m_inputData.GetScalarOpt("waitTime");                // get time of waiting
    // pull_speed = m_inputData.GetScalarOpt("pullSpeed");              // get speed of pulling
        
    col_limit = m_inputData.GetScalarOpt("colLimit");                // distance limit for candidate set
    delta = m_inputData.GetScalarOpt("delta");                       // distance tolerance for contact
    k_scaler = m_inputData.GetScalarOpt("kScaler");                  // constant scaler for contact stiffness
    mu = m_inputData.GetScalarOpt("mu");                             // friction coefficient
    nu = m_inputData.GetScalarOpt("nu");                             // slipping tolerance for friction
    line_search = m_inputData.GetIntOpt("lineSearch");               // flag for enabling line search
    // knot_config = m_inputData.GetStringOpt("knotConfig");            // get initial knot configuration

    shearM = youngM / (2.0 * (1.0 + Poisson));                             // shear modulus

    alpha = 1.0;                                                           // newton step size

    point_vel = Vector3d::Zero();

    deltaLength = (RodLength / (numVertices - 1));
}

world::~world() {
    ;
}

void world::setRodStepper() {
    // Set up geometry
    rodGeometry();

    // Create the rod
    rod = new elasticRod(vertices, vertices, density, rodRadius, deltaTime,
                         youngM, shearM, RodLength, theta);

    // Find out the tolerance, e.g. how small is enough?
    characteristicForce = M_PI * pow(rodRadius, 4) / 4.0 * youngM / pow(RodLength, 2);
    forceTol = tol * characteristicForce;

    // Set up boundary condition
    rodBoundaryCondition();

    // setup the rod so that all the relevant variables are populated
    rod->setup();
    // End of rod setup

    // set up the time stepper
    stepper = new timeStepper(*rod);
    totalForce = stepper->getForce();
    ls_nodes = new double[rod->ndof];
    dx = stepper->dx;

    // declare the forces
    m_stretchForce = new elasticStretchingForce(*rod, *stepper);
    // m_bendingForce = new elasticBendingForce(*rod, *stepper);
    // m_twistingForce = new elasticTwistingForce(*rod, *stepper);
    m_inertialForce = new inertialForce(*rod, *stepper);
    m_gravityForce = new externalGravityForce(*rod, *stepper, gVector);
    m_dampingForce = new dampingForce(*rod, *stepper, viscosity);
    m_collisionDetector = new collisionDetector(*rod, delta, col_limit);
    m_contactPotentialIMC = new contactPotentialIMC(*rod, *stepper, *m_collisionDetector, delta, k_scaler, mu, nu);

    // Allocate every thing to prepare for the first iteration
    rod->updateTimeStep();

    currentTime = 0.0;
    timeStep = 0;
}

// Setup geometry
void world::rodGeometry() {

    vertices = MatrixXd(numVertices, 3);

    theta = VectorXd::Zero(numVertices - 1);

    for (int i = 0; i < numVertices; i++) {
        vertices(i, 0) = deltaLength*i;
        vertices(i, 1) = 0.0;
        vertices(i, 2) = 0.0;
    }
}

void world::rodBoundaryCondition() {
    // rod -> setVertexPlanarBoundaryCondition(1);

    rod->setVertexBoundaryCondition(rod->getVertex(0), 0);
    rod->setVertexBoundaryCondition(rod->getVertex(1), 1);
    rod->setThetaBoundaryCondition(0.0, 0);
    // rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 1), numVertices - 1);
    // rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 2), numVertices - 2);
    // rod->setThetaBoundaryCondition(0.0, numVertices - 2);
}

void world::updateBoundary() {
    Vector3d u;
    u(0) = point_vel(0);
    u(1) = 0;
    u(2) = point_vel(1);

    rod->setVertexBoundaryCondition(rod->getVertex(0) + u * deltaTime, 0);

    Eigen::VectorXd q0 = rod->getVertex(0);
    Eigen::VectorXd q1 = rod->getVertex(1);

    Eigen::VectorXd q_(2, 1);
    q_(0, 0) =  q0(0) - q1(0);
    q_(1, 0) =  q0(2) - q1(2);

    Eigen::MatrixXd rot(2, 2);
    rot(0, 0) = cos(M_PI / 2);
    rot(0, 1) = -sin(M_PI / 2);
    rot(1, 0) = sin(M_PI / 2);
    rot(1, 1) = cos(M_PI / 2);

    VectorXd u_(2, 1);
    u_(0) = point_vel(0);
    u_(1) = point_vel(1);

    u_ = point_vel(2) * rot * q_ / q_.norm() * deltaLength + u_;
    
    Vector3d q_new;
    q_new(0) = q1(0) + u_(0) * deltaTime;
    q_new(1) = 0;    
    q_new(2) = q1(2) + u_(1) * deltaTime;
    
    // maintain distance constraint            
    q_new = q_new / q_new.norm() * deltaLength;

    rod->setVertexBoundaryCondition(q_new, 1);
}

void world::updateCons() {
    rod->updateMap();
    stepper->update();
    totalForce = stepper->getForce();
}

void world::updateTimeStep() {
    bool solved = false;

    updateBoundary();

    newtonMethod(solved);

    // calculate pull forces;
    calculateForce();

    rod->updateTimeStep();

    // printSimData();

}

void world::calculateForce() {
    stepper->setZero();

    m_inertialForce->computeFi();
    m_stretchForce->computeFs();
    // m_bendingForce->computeFb();
    // m_twistingForce->computeFt();
    m_gravityForce->computeFg();
    m_dampingForce->computeFd();

    temp[0] = stepper->force[0] + stepper->force[4];
    temp[1] = stepper->force[1] + stepper->force[5];
    temp[2] = stepper->force[2] + stepper->force[6];

    temp1[0] = stepper->force[rod->ndof - 3] + stepper->force[rod->ndof - 7];
    temp1[1] = stepper->force[rod->ndof - 2] + stepper->force[rod->ndof - 6];
    temp1[2] = stepper->force[rod->ndof - 1] + stepper->force[rod->ndof - 5];
}

void world::newtonDamper() {
    if (iter < 10)
        alpha = 1.0;
    else
        alpha *= 0.90;
    if (alpha < 0.1)
        alpha = 0.1;
    // std::cout << alpha << std::endl;
    // alpha = 1.0;
}

void world::newtonMethod(bool &solved) {
    double normf = forceTol * 10.0;
    double normf0 = 0;
    iter = 0;

    double curr_weight = 1.0;
    int counter = 0;
    while (true) {
        rod->updateGuess(curr_weight);
        if (m_collisionDetector->constructCandidateSet(counter > 10)) break;
        curr_weight /= 2;
        counter++;
    }

    while (solved == false) {
        rod->prepareForIteration();

        stepper->setZero();

        // Compute the forces and the jacobians
        m_inertialForce->computeFi();
        m_inertialForce->computeJi();

        m_stretchForce->computeFs();
        m_stretchForce->computeJs();

        // m_bendingForce->computeFb();
        // m_bendingForce->computeJb();

        // m_twistingForce->computeFt();
        // m_twistingForce->computeJt();

        m_gravityForce->computeFg();
        m_gravityForce->computeJg();

        m_dampingForce->computeFd();
        m_dampingForce->computeJd();

        m_collisionDetector->detectCollisions();
        if (iter == 0) {
            m_contactPotentialIMC->updateContactStiffness();
        }

        m_contactPotentialIMC->computeFcJc();

        // Compute norm of the force equations.
        normf = 0;
        for (int i = 0; i < rod->uncons; i++) {
            normf += totalForce[i] * totalForce[i];
        }
        normf = sqrt(normf);


        if (iter == 0) {
            normf0 = normf;
        }

        if (normf <= forceTol || (iter > 0 && normf <= normf0 * stol)) {
            solved = true;
            iter++;
        //     if (pulling())
        //         total_iters++;
        }

        if (solved == false) {
            stepper->integrator(); // Solve equations of motion
            if (line_search) {
                lineSearch();
            } else {
                newtonDamper();
            }
            rod->updateNewtonX(dx, alpha); // new q = old q + Delta q
            iter++;
        //     if (pulling())
        //         total_iters++;
        }

        // Exit if unable to converge
        if (iter > maxIter) {
            cout << "No convergence after " << maxIter << " iterations" << endl;
            exit(1);
        }
    }
}

void world::lineSearch() {
    // store current x
    rod->xold = rod->x;
    // Initialize an interval for optimal learning rate alpha
    double amax = 2;
    double amin = 1e-3;
    double al = 0;
    double au = 1;

    double a = 1;

    //compute the slope initially
    double q0 = 0.5 * pow(stepper->Force.norm(), 2);
    double dq0 = -(stepper->Force.transpose() * stepper->Jacobian * stepper->DX)(0);

    bool success = false;
    double m2 = 0.9;
    double m1 = 0.1;
    int iter_l = 0;
    while (!success) {
        rod->x = rod->xold;
        rod->updateNewtonX(dx, a);

        rod->prepareForIteration();

        stepper->setZero();

        // Compute the forces and the jacobians
        m_inertialForce->computeFi();
        m_stretchForce->computeFs();
        // m_bendingForce->computeFb();
        // m_twistingForce->computeFt();
        m_gravityForce->computeFg();
        m_dampingForce->computeFd();
        m_collisionDetector->detectCollisions();
        m_contactPotentialIMC->computeFc();

        double q = 0.5 * pow(stepper->Force.norm(), 2);

        double slope = (q - q0) / a;

        if (slope >= m2 * dq0 && slope <= m1 * dq0) {
            success = true;
        }
        else {
            if (slope < m2 * dq0) {
                al = a;
            }
            else {
                au = a;
            }

            if (au < amax) {
                a = 0.5 * (al + au);
            }
            else {
                a = 10 * a;
            }
        }
        if (a > amax || a < amin) {
            break;
        }
        if (iter_l > 100) {
            break;
        }
        iter_l++;
    }
    rod->x = rod->xold;

    alpha = a;
}

const double world::getCurrentTime() {return currentTime;}

const Eigen::VectorXd world::getStatePos() {return rod->x;}

const Eigen::VectorXd world::getStateVel() {return rod->u;}

const Eigen::MatrixXd world::getStateD1() {return rod->d1;}

const Eigen::MatrixXd world::getStateD2() {return rod->d2;}

const Eigen::MatrixXd world::getStateTangent() {return rod->tangent;}

const Eigen::VectorXd world::getStateRefTwist() {return rod->refTwist;}


void world::setStatePos(Eigen::VectorXd x) {rod->x0 = x; rod->x = x;}

void world::setStateVel(Eigen::VectorXd u) {rod->u = u;}

void world::setStateD1(Eigen::MatrixXd d1) {rod->d1 = d1; rod->d1_old = d1;}

void world::setStateD2(Eigen::MatrixXd d2) {rod->d2 = d2; rod->d2_old = d2;}

void world::setStateTangent(Eigen::MatrixXd tangent) {rod->tangent = tangent; rod->tangent_old = tangent;}

void world::setStateRefTwist(Eigen::VectorXd refTwist) {rod->refTwist = refTwist; rod->refTwist_old = refTwist;}


void world::setPointVel(Eigen::Vector3d u) {point_vel = u;}
