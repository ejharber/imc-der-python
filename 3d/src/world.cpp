#include "world.h"

world::world(string name, Eigen::MatrixXd vertices_, Eigen::MatrixXd vertices_home_, double radius) {

    setInput m_inputData;
    m_inputData = setInput();
    m_inputData.LoadOptions(name);

    // simulation parameters
    gVector = m_inputData.GetVecOpt("gVector");                            // m/s^2
    maxIter = m_inputData.GetIntOpt("maxIter");                      // maximum number of iterations
    line_search = m_inputData.GetIntOpt("lineSearch");               // flag for enabling line search
    alpha = 1.0;                                                           // newton step size

    rodRadius = radius;               // meter

    numVertices = vertices_.rows();
    
    vertices = vertices_;
    theta = VectorXd::Zero(numVertices - 1);

    vertices_home = vertices_home_;
    theta_home = VectorXd::Zero(numVertices - 1);

    RodLength = 0;               // meter

    for (int i = 0; i < numVertices-1; i++) {
        Vector3d vec = vertices_home.row(i) - vertices_home.row(i + 1);
        RodLength += vec.norm();
    }

    // physics parameters
    youngM = m_inputData.GetScalarOpt("youngM");                     // Pa
    Poisson = m_inputData.GetScalarOpt("Poisson");                   // dimensionless
    deltaTime = m_inputData.GetScalarOpt("deltaTime");               // seconds
    tol = m_inputData.GetScalarOpt("tol");                           // small number like 10e-7
    stol = m_inputData.GetScalarOpt("stol");                         // small number, e.g. 0.1%
    density = m_inputData.GetScalarOpt("density");                   // kg/m^3
    viscosity = m_inputData.GetScalarOpt("viscosity");               // viscosity in Pa-s   
    col_limit = m_inputData.GetScalarOpt("colLimit");                // distance limit for candidate set
    delta = m_inputData.GetScalarOpt("delta");                       // distance tolerance for contact
    k_scaler = m_inputData.GetScalarOpt("kScaler");                  // constant scaler for contact stiffness
    mu = m_inputData.GetScalarOpt("mu");                             // friction coefficient
    nu = m_inputData.GetScalarOpt("nu");                             // slipping tolerance for friction
    shearM = youngM / (2.0 * (1.0 + Poisson));                             // shear modulus

    point_vel = Vector3d::Zero();

}

world::~world() {
    ;
}

void world::setRodStepper() {
    // Set up geometry

    // Create the rod
    rod = new elasticRod(vertices, vertices_home, density, rodRadius, deltaTime,
                         youngM, shearM, RodLength, theta_home);

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
    m_bendingForce = new elasticBendingForce(*rod, *stepper);
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

    Eigen::VectorXd q0 = rod->getVertex(0); // vertex of node 0
    Eigen::VectorXd q1 = rod->getVertex(1); // vertex of node 1

    Vector3d u;
    u(0) = point_vel(0);
    u(1) = 0;
    u(2) = point_vel(1);

    rod->setVertexBoundaryCondition(q0 + u * deltaTime, 0);

    Eigen::VectorXd q_ = q0 - q1;

    Eigen::MatrixXd rot(3, 3);
    rot(0, 0) = 0.0;
    rot(0, 1) = 0.0;
    rot(0, 2) = sin(M_PI / 2);

    rot(1, 0) = 0.0;
    rot(1, 1) = 1.0;
    rot(1, 2) = 0.0;

    rot(2, 0) = -sin(M_PI / 2);
    rot(2, 1) = 0.0;
    rot(2, 2) = 0.0;

    float dl = (vertices_home.row(0) - vertices_home.row(1)).norm();

    u += point_vel(2) * rot * q_ / q_.norm() * dl;
    
    Vector3d q_new = q1 + u * deltaTime;

    // maintain distance constraint         
    Vector3d q_norm = q_new - (rod->getVertex(0) + u * deltaTime);
    q_new = q_norm / q_norm.norm() * dl + (rod->getVertex(0) + u * deltaTime);

    rod->setVertexBoundaryCondition(q_new, 1);
}

void world::updateCons() {
    rod->updateMap();
    stepper->update();
    totalForce = stepper->getForce();
}

bool world::updateTimeStep() {
    bool solved = false;

    updateBoundary();

    newtonMethod(solved);

    if (!solved) {
        return false;
    }

    // calculate pull forces;
    calculateForce();

    rod->updateTimeStep();

    return solved;
}

void world::calculateForce() {
    stepper->setZero();

    m_inertialForce->computeFi();
    m_stretchForce->computeFs();
    m_bendingForce->computeFb();
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
    alpha = 1.0;
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

        m_bendingForce->computeFb();
        m_bendingForce->computeJb();

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
        }

        // Exit if unable to converge
        // this should not exit!
        if (iter > maxIter) {
            cout << "No convergence after " << maxIter << " iterations" << endl;
            // return false;
            // exit(1);
            break;
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
        m_bendingForce->computeFb();
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
