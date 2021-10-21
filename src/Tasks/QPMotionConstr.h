/*
 * Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

// includes
// std
#include <map>

// Eigen
#include <Eigen/Core>

// RBDyn
#include <RBDyn/FD.h>
#include <RBDyn/Jacobian.h>

// Tasks
#include "QPSolver.h"

namespace tasks
{
struct TorqueBound;
struct TorqueDBound;
struct PolyTorqueBound;

namespace qp
{

class TASKS_DLLAPI PositiveLambda : public ConstraintFunction<Bound>
{
public:
  PositiveLambda();

  // Constraint
  virtual void updateNrVars(const std::vector<rbd::MultiBody> & mbs, const SolverData & data) override;
  virtual void update(const std::vector<rbd::MultiBody> & mbs,
                      const std::vector<rbd::MultiBodyConfig> & mbc,
                      const SolverData & data) override;

  // Description
  virtual std::string nameBound() const override;
  virtual std::string descBound(const std::vector<rbd::MultiBody> & mbs, int line) override;

  // Bound Constraint
  virtual int beginVar() const override;

  virtual const Eigen::VectorXd & Lower() const override;
  virtual const Eigen::VectorXd & Upper() const override;

private:
  struct ContactData
  {
    ContactId cId;
    int lambdaBegin, nrLambda; // lambda index in x
  };

private:
  int lambdaBegin_;
  Eigen::VectorXd XL_, XU_;

  std::vector<ContactData> cont_; // only usefull for descBound
};

class TASKS_DLLAPI MotionConstrCommon : public ConstraintFunction<GenInequality>
{
public:
  MotionConstrCommon(const std::vector<rbd::MultiBody> & mbs, int robotIndex);

  void computeTorque(const Eigen::VectorXd & alphaD, const Eigen::VectorXd & lambda);
  const Eigen::VectorXd & torque() const;
  void torque(const std::vector<rbd::MultiBody> & mbs, std::vector<rbd::MultiBodyConfig> & mbcs) const;

  // Constraint
  virtual void updateNrVars(const std::vector<rbd::MultiBody> & mbs, const SolverData & data) override;

  void computeMatrix(const std::vector<rbd::MultiBody> & mb, const std::vector<rbd::MultiBodyConfig> & mbcs);

  // Description
  virtual std::string nameGenInEq() const override;
  virtual std::string descGenInEq(const std::vector<rbd::MultiBody> & mbs, int line) override;

  // Inequality Constraint
  virtual int maxGenInEq() const override;

  virtual const Eigen::MatrixXd & AGenInEq() const override;
  virtual const Eigen::VectorXd & LowerGenInEq() const override;
  virtual const Eigen::VectorXd & UpperGenInEq() const override;

protected:
  struct ContactData
  {
    ContactData() {}
    ContactData(const rbd::MultiBody & mb,
                const std::string & bodyName,
                int lambdaBegin,
                std::vector<Eigen::Vector3d> points,
                const std::vector<FrictionCone> & cones);

    int bodyIndex;
    int lambdaBegin;
    rbd::Jacobian jac;
    std::vector<Eigen::Vector3d> points;
    // BEWARE generator are minus to avoid one multiplication by -1 in the
    // update method
    std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> minusGenerators;
  };

protected:
  int robotIndex_, alphaDBegin_, nrDof_, lambdaBegin_;
  rbd::ForwardDynamics fd_;
  Eigen::MatrixXd fullJacLambda_, jacTrans_, jacLambda_;
  std::vector<ContactData> cont_;

  Eigen::VectorXd curTorque_;

  Eigen::MatrixXd A_;
  Eigen::VectorXd AL_, AU_;
  size_t updateIter_ = 0;
};

/**
 * Avoid reaching torque limits based on a linear velocity damper.
 *
 * A linear velocity damper equation writes as
 * \f[
 * \dot{d}^k \geq - \xi \frac{d^k - d_s}{d_i - d_s}
 * \f]
 * where \f$ d^k \f$ denotes the distance to a limit at iteration \f$ k \f$.
 *
 * \f$ d_i \f$ is an interaction distance constant. The linear velocity damper constraint is activated only when \f$ d^k < d_i \f$.
 *
 * \f$ d_s \f$ is a safety distance constant, and \f$ \xi \f$ is a damping constant, computed when constraint is activated as
 * \f[
 * \xi = - \frac{d_i - d_s}{d^k - d_s} \dot{d^k} + \xi_\text{off}
 * \f]
 * with \f$ \xi_\text{off} \f$ a user-specified constant (usually \f$ 0.5 \f$).
 *
 * A constraint written in this form allows to reduce the state speed \f$ \dot{d}^k \f$ when approaching the state limit, until \f$ \dot{d}^k \f$ becomes zero when \f$ d^k = d_s \f$.
 *
 *
 * In case of torque limits two following distances are considered
 * \f[
 * \underline{d} = \tau - \underline{\tau}
 * \f]
 * \f[
 * \overline{d} = \overline{\tau} - \tau
 * \f]
 * with their respective derivatives
 * \f[
 * \dot{\underline{d}} = \dot{\tau}
 * \f]
 * \f[
 * \dot{\overline{d}} = - \dot{\tau}
 * \f]
 * assuming that the torque limits are constant.
 *
 * The corresponding linear velocity damper constraints are
 * \f[
 * \dot{\tau} \geq -\xi_l \frac{\underline{d}^k - d_s}{d_i - d_s}
 * \f]
 * \f[
 * \xi_l = - \frac{d_i - d_s}{\underline{d}^k - d_s} \dot{\tau} + \xi_\text{off} \text{, where } \dot{\tau} \text{ is computed numerically as } \frac{\tau_{k-1} - \tau_{k-2}}{\Delta_{dt}}
 * \f]
 * and
 * \f[
 * -\dot{\tau} \geq -\xi_u \frac{\overline{d}^k - d_s}{d_i - d_s}
 * \f]
 * \f[
 * \xi_u = - \frac{d_i - d_s}{\overline{d}^k - d_s} (-\dot{\tau}) + \xi_\text{off} \text{, where } \dot{\tau} \text{ is computed numerically as } \frac{\tau_{k-1} - \tau_{k-2}}{\Delta_{dt}}
 * \f]
 * These constraints combined impose lower and upper limit on the torque derivative
 * \f[
 * -\xi_l \frac{\underline{d}^k - d_s}{d_i - d_s} \leq \dot{\tau} \leq \xi_u \frac{\overline{d}^k - d_s}{d_i - d_s}
 * \f]
 * which translate into the following constraint on joints acceleration \f$ \ddot{q} \f$ and contact forces \f$ f \f$
 * \f[
 * \tau_{k-1} + (-\xi_l \frac{\underline{d}^k - d_s}{d_i - d_s})\Delta_{dt} - C \leq M(q)\ddot{q} - J^T f \leq \tau_{k-1} + \xi_u \frac{\overline{d}^k - d_s}{d_i - d_s} \Delta_{dt} - C
 * \f]
 * In order to account also for the torque limits (\f$ \underline{\tau} \f$ and \f$ \overline{\tau} \f$) and default torque derivative limits (\f$ \underline{\dot{\tau}} \f$ and \f$ \overline{\dot{\tau}} \f$) the final constraint takes on the following form
 * \f[
 * \max(\underline{\tau}, \tau_{k-1} + \underline{\dot{\tau}}\Delta_{dt}, \tau_{k-1} + (-\xi_l \frac{\underline{d}^k - d_s}{d_i - d_s})\Delta_{dt}) - C \leq M(q)\ddot{q} - J^T f \leq \min(\overline{\tau}, \tau_{k-1} + \overline{\dot{\tau}}\Delta_{dt}, \tau_{k-1} + \xi_u \frac{\overline{d}^k - d_s}{d_i - d_s} \Delta_{dt}) - C
 * \f]
 */
class TASKS_DLLAPI MotionConstr : public MotionConstrCommon
{
public:
  MotionConstr(const std::vector<rbd::MultiBody> & mbs, int robotIndex, const TorqueBound & tb);

  MotionConstr(const std::vector<rbd::MultiBody> & mbs,
               int robotIndex,
               const TorqueBound & tb,
               const TorqueDBound & tdb,
               double dt);

  // Constraint
  virtual void update(const std::vector<rbd::MultiBody> & mbs,
                      const std::vector<rbd::MultiBodyConfig> & mbcs,
                      const SolverData & data) override;
  // Matrix
  const Eigen::MatrixXd matrix() const
  {
    return A_;
  }
  // Contact torque
  Eigen::MatrixXd contactMatrix() const;
  // Access fd...
  const rbd::ForwardDynamics fd() const;

protected:
  Eigen::VectorXd torqueL_, torqueU_;
  Eigen::VectorXd torqueDtL_, torqueDtU_;
  Eigen::VectorXd tmpL_, tmpU_;

private:
  struct DampData
  {
    enum State
    {
      Low,
      Upp,
      Free
    };

    DampData(double mi, double ma, double idi, double sdi, int aDB, int i)
    : min(mi), max(ma), iDist(idi), sDist(sdi), jointIndex(i), alphaDBegin(aDB), damping(0.),
      state(Free)
    {
    }

    double min, max;
    double iDist, sDist;
    int jointIndex;
    int alphaDBegin;
    double damping;
    State state;
  };

  std::vector<DampData> data_;
  std::vector<std::vector<double>> prevJointTorque_;
  double dt_;
  double damperOff_ = 0.5;
  Eigen::VectorXd dampedTorqueDtL_, dampedTorqueDtU_;

};

struct SpringJoint
{
  SpringJoint() {}
  SpringJoint(const std::string & jName, double K, double C, double O) : jointName(jName), K(K), C(C), O(O) {}

  std::string jointName;
  double K, C, O;
};

class TASKS_DLLAPI MotionSpringConstr : public MotionConstr
{
public:
  MotionSpringConstr(const std::vector<rbd::MultiBody> & mbs,
                     int robotIndex,
                     const TorqueBound & tb,
                     const std::vector<SpringJoint> & springs);

  MotionSpringConstr(const std::vector<rbd::MultiBody> & mbs,
                     int robotIndex,
                     const TorqueBound & tb,
                     const TorqueDBound & tdb,
                     double dt,
                     const std::vector<SpringJoint> & springs);

  // Constraint
  virtual void update(const std::vector<rbd::MultiBody> & mbs,
                      const std::vector<rbd::MultiBodyConfig> & mbc,
                      const SolverData & data) override;

protected:
  struct SpringJointData
  {
    int index;
    int posInDof;
    double K;
    double C;
    double O;
  };

protected:
  std::vector<SpringJointData> springs_;
};

/**
 * @brief Use polynome in function of q to compute torque limits.
 * BEWARE: Only work with 1 dof/param joint
 */
class TASKS_DLLAPI MotionPolyConstr : public MotionConstrCommon
{
public:
  MotionPolyConstr(const std::vector<rbd::MultiBody> & mbs, int robotIndex, const PolyTorqueBound & ptb);

  // Constraint
  virtual void update(const std::vector<rbd::MultiBody> & mbs,
                      const std::vector<rbd::MultiBodyConfig> & mbcs,
                      const SolverData & data) override;

protected:
  std::vector<Eigen::VectorXd> torqueL_, torqueU_;
  std::vector<int> jointIndex_;
};

} // namespace qp

} // namespace tasks
