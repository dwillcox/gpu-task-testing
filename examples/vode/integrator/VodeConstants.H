#ifndef _VODECONSTANTS_H
#define _VODECONSTANTS_H
#include <AMReX_REAL.H>
#include <AMReX_CONSTANTS.H>

#define ADAMS_Q_MAX 12     /* max value of q for lmm == ADAMS     */
#define BDF_Q_MAX    5     /* max value of q for lmm == BDF       */
#define Q_MAX  ADAMS_Q_MAX /* max value of q for either lmm       */
#define L_MAX  (Q_MAX+1)   /* max value of L for either lmm       */
#define NUM_TESTS    5     /* number of error test quantities     */

#define HMIN_DEFAULT     zero           /* hmin default value     */
#define HMAX_INV_DEFAULT zero           /* hmax_inv default value */
#define MXHNIL_DEFAULT   10             /* mxhnil default value   */
#define MXSTEP_DEFAULT   500            /* mxstep default value   */

/*
 * Algorithmic constants
 * ---------------------
 *
 * CVodeGetDky and cvStep
 *
 *    FUZZ_FACTOR
 *
 * cvHin
 *
 *    HLB_FACTOR
 *    HUB_FACTOR
 *    H_BIAS
 *    MAX_ITERS
 *
 * CVodeCreate 
 *
 *   CORTES
 *
 * cvStep
 *
 *    THRESH
 *    ETAMX1
 *    ETAMX2
 *    ETAMX3
 *    ETAMXF
 *    ETAMIN
 *    ETACF
 *    ADDON
 *    BIAS1
 *    BIAS2
 *    BIAS3
 *    ONEPSM
 *
 *    SMALL_NST   nst > SMALL_NST => use ETAMX3 
 *    MXNCF       max no. of convergence failures during one step try
 *    MXNEF       max no. of error test failures during one step try
 *    MXNEF1      max no. of error test failures before forcing a reduction of order
 *    SMALL_NEF   if an error failure occurs and SMALL_NEF <= nef <= MXNEF1, then
 *                reset eta =  SUNMIN(eta, ETAMXF)
 *    LONG_WAIT   number of steps to wait before considering an order change when
 *                q==1 and MXNEF1 error test failures have occurred
 *
 * cvNls
 *
 *    DGMAX       iter == CV_NEWTON, |gamma/gammap-1| > DGMAX => call lsetup
 *    MSBP        max no. of steps between lsetup calls
 *
 */

#define FUZZ_FACTOR 100.0

#define HLB_FACTOR 100.0
#define HUB_FACTOR tenth
#define H_BIAS     half
#define MAX_ITERS  4

#define CORTES tenth

#define THRESH 1.5
#define ETAMX1 10000.0 
#define ETAMX2 ten
#define ETAMX3 ten
#define ETAMXF fifth
#define ETAMIN tenth
#define ETACF  fourth
#define ADDON  0.000001
#define BIAS1  six
#define BIAS2  six
#define BIAS3  ten
#define ONEPSM 1.000001

#define SMALL_NST    10
#define MXNCF        10
#define MXNEF         7
#define MXNEF1        3
#define SMALL_NEF     2
#define LONG_WAIT    10

#define DGMAX  0.3
#define MSBP   20

enum class VodeStepping {Normal = 1, OneStep = 2};
enum class MultiStepMethod {Adams = 1, BDF = 2};
enum class ToleranceType {NoneNone = 0, ScalarScalar = 1, ScalarVector = 2, WeightFunction = 3};
enum class CorrectorConvergence {NoFailures = 0, BadJacFailure = 1, OtherFailure = 2};
enum class VodeState {InternalError,
                      PrepareFirstStep,
                      EstimateStartingTimestep, EstimateStartingTimestep_RHS,
                      ApplyStartingTimestep,
                      AdvanceMultistep,
                      CallRightHandSide,
                      SolveLinearSystem,
                      IntegrationComplete};
enum class InternalErrorState {None, InvalidToleranceType, InvalidTimestepSign, TargetIntervalTooSmall,
        InvalidInterpolantDerivativeOrderRequested, InvalidInterpolantTime,
        ExceededMaximumSteps, RequestedTooMuchAccuracy,
        ExceededMaximumNilSteps, BetterPredictionNeeded, HardCorrectorFailure};

enum class InternalWarningState {None, UsingNilStepsize};

enum class MultiStepCheckpoint {Prepare, Execute};
enum class SingleStepCheckpoint {Prepare, PredictAndSet, SolveNonlinear, EvaluateNonlinear, FinishStep};
enum class NonlinearCheckpoint {Prepare, Execute, Evaluate};
enum class NewtonCheckpoint {Prepare, Execute, Evaluate};

#endif
