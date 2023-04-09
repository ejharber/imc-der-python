#ifndef WORLD_H
#define WORLD_H

#include "eigenIncludes.h"

// include elastic rod class
#include "elasticRod.h"

// include force classes
#include "elasticStretchingForce.h"
#include "elasticBendingForce.h"
#include "elasticTwistingForce.h"
#include "externalGravityForce.h"
#include "inertialForce.h"

// include external force
#include "dampingForce.h"
#include "contactPotentialIMC.h"

// include time stepper
#include "timeStepper.h"

// include input file and option
#include "setInput.h"
#include <math.h>

class world
{
public:
    world(string name, Eigen::MatrixXd vertices_, Eigen::MatrixXd vertices_home_, double radius);
    ~world();
    void setRodStepper();
    bool updateTimeStep();

    const double getCurrentTime();
    
    const Eigen::VectorXd getStatePos();
    const Eigen::VectorXd getStateVel();
    const Eigen::MatrixXd getStateD1();
    const Eigen::MatrixXd getStateD2();
    const Eigen::MatrixXd getStateTangent();
    const Eigen::VectorXd getStateRefTwist();

    void setStatePos(Eigen::VectorXd x);
    void setStateVel(Eigen::VectorXd u);
    void setStateD1(Eigen::MatrixXd d1_old);
    void setStateD2(Eigen::MatrixXd d2_old);
    void setStateTangent(Eigen::MatrixXd tangent_old);
    void setStateRefTwist(Eigen::VectorXd refTwist_old);

    void setPointVel(Eigen::Vector3d u);

private:

    // Physical parameters
    double RodLength;
    double rodRadius;
    int numVertices;
    // double deltaLength;
    double youngM;
    double Poisson;
    double shearM;
    double deltaTime;
    double density;
    Vector3d gVector;
    double viscosity;
    double col_limit;
    double delta;
    double k_scaler;
    double mu;
    double nu;
    int line_search;
    double alpha;

    double tol, stol;
    int maxIter; // maximum number of iterations
    double characteristicForce;
    double forceTol;

    // Geometry
    MatrixXd vertices;
    VectorXd theta;

    MatrixXd vertices_home;
    VectorXd theta_home;

    // set up the time stepper
    timeStepper *stepper;
    double *totalForce;
    double *dx;
    double *ls_nodes;
    double currentTime;
    int timeStep;
    double totalTime;

    // Rod
    elasticRod *rod;

    // declare the forces
    elasticStretchingForce *m_stretchForce;
    elasticBendingForce *m_bendingForce;
    // elasticTwistingForce *m_twistingForce;
    inertialForce *m_inertialForce;
    externalGravityForce *m_gravityForce;
    dampingForce *m_dampingForce;
    collisionDetector *m_collisionDetector;
    contactPotentialIMC *m_contactPotentialIMC;

    int iter;

    void rodBoundaryCondition();

    void updateBoundary();

    void updateCons();

    void newtonMethod(bool &solved);
    void newtonDamper();
    void calculateForce();
    void lineSearch();

    Vector3d temp;
    Vector3d temp1;

    Vector3d gravity;
    Vector3d inertial;
    Vector3d dampingF;

    Vector3d point_vel;
};

#endif
