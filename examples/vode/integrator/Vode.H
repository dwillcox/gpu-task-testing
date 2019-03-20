#ifndef _VODE_H
#define _VODE_H
#include <cassert>
#include "unified.H"
#include "RealVector.H"
#include "VodeConstants.H"

using namespace amrex;

template<size_t N> class Vode : public UnifiedMemoryClass {
    // Current State of Vode execution
    VodeState current_state;
    VodeState previous_state;
    InternalErrorState error_state;
    InternalWarningState warning_state;

    CheckpointContainer<MultiStepCheckpoint, 2> multistep_checkpoints;
    CheckpointContainer<SingleStepCheckpoint, 3> singlestep_checkpoints;
    CheckpointContainer<NonlinearCheckpoint, 3> nonlinear_checkpoints;
    CheckpointContainer<NewtonCheckpoint, 3> newton_checkpoints;

    // single step data
    Real saved_time; // saved_t
    Real weighted_step_error; // dsm
    long int number_step_corrector_failures; // ncf
    long int number_step_error_failures; // nef
    // end single step data

    Real cv_uround;
    void* cv_user_data;

    MultiStepMethod cv_lmm;
    ToleranceType cv_itol;

    Real cv_reltol;
    Real cv_Sabstol;
    RealVector<N> cv_Vabstol;

    // skip allowing a user error weight function

    RealVector<N> cv_zn[L_MAX];
    RealVector<N> cv_ewt;
    RealVector<N> cv_acor;

    RealVector<N> cv_ytmp;
    RealVector<N> cv_ydot;

    VodeStepping vode_stepping;

    Real target_time;

    Real time_roundoff;

    // needed by starting timestep init    
    Real start_h_upper_bound;
    Real start_h_lower_bound;
    Real esthin_h_previous;

    // needed to resume a loop
    int state_resume_counter;

    // Vode stuff ...
    int cv_q;                    /* current order                               */
    int cv_qprime;               /* order to be used on the next step  
                                    = q-1, q, or q+1                            */
    int cv_next_q;               /* order to be used on the next step           */
    int cv_qwait;                /* number of internal steps to wait before
                                    considering a change in q                   */
    int cv_L;                    /* L = q + 1                                   */

    Real cv_hin;             /* initial step size                           */
    Real cv_h;               /* current step size                           */
    Real cv_hprime;          /* step size to be used on the next step       */ 
    Real cv_next_h;          /* step size to be used on the next step       */ 
    Real cv_eta;             /* eta = hprime / h                            */
    Real cv_hscale;          /* value of h used in zn                       */
    Real cv_tn;              /* current internal value of t                 */
    Real cv_tretlast;        /* value of tret last returned by CVode        */

    Real cv_tau[L_MAX+1];    /* array of previous q+1 successful step 
                                    sizes indexed from 1 to q+1                 */
    Real cv_tq[NUM_TESTS+1]; /* array of test quantities indexed from 
                                    1 to NUM_TESTS(=5)                          */
    Real cv_l[L_MAX];        /* coefficients of l(x) (degree q poly)        */

    Real cv_rl1;              /* the scalar 1/l[1]                          */
    Real cv_gamma;            /* gamma = h * rl1                            */
    Real cv_gammap;           /* gamma at the last setup call               */
    Real cv_gamrat;           /* gamma / gammap                             */

    Real cv_crate;            /* estimated corrector convergence rate       */
    Real cv_acnrm;            /* | acor | wrms                              */
    Real cv_nlscoef;          /* coeficient in nonlinear convergence test   */

    /*------
      Limits 
      ------*/

    int cv_qmax;          /* q <= qmax                                          */
    long int cv_mxstep;   /* maximum number of internal steps for one user call */
    int cv_maxcor;        /* maximum number of corrector iterations for the
                             solution of the nonlinear equation                 */
    int cv_mxhnil;        /* maximum number of warning messages issued to the
                             user that t + h == t for the next internal step    */
    int cv_maxnef;        /* maximum number of error test failures              */
    int cv_maxncf;        /* maximum number of nonlinear convergence failures   */

    Real cv_hmin;     /* |h| >= hmin                                        */
    Real cv_hmax_inv; /* |h| <= 1/hmax_inv                                  */
    Real cv_etamax;   /* eta <= etamax                                      */

    /*--------
      Counters 
      --------*/

    long int num_steps;         /* number of internal steps taken                  */ // (cv_nst)
    long int cv_nfe;         /* number of f calls                               */
    long int cv_ncfn;        /* number of corrector convergence failures        */
    long int cv_netf;        /* number of error test failures                   */
    long int cv_nni;         /* number of Newton iterations performed           */
    long int cv_nsetups;     /* number of setup calls                           */
    int cv_nhnil;            /* number of messages issued to the user that 
                                t + h == t for the next iternal step            */

    Real cv_etaqm1;      /* ratio of new to old h for order q-1             */
    Real cv_etaq;        /* ratio of new to old h for order q               */
    Real cv_etaqp1;      /* ratio of new to old h for order q+1             */

    /*------------
      Saved Values
      ------------*/

    int cv_qu;                   /* last successful q value used                */
    long int cv_nstlp;           /* step number of last setup call              */
    Real cv_h0u;                 /* actual initial stepsize                     */
    Real cv_hu;                  /* last successful h value used                */
    Real cv_saved_tq5;           /* saved value of tq[5]                        */
    bool cv_jcur;                /* is Jacobian info. for lin. solver current?  */
    Real cv_tolsf;               /* tolerance scale factor                      */
    int cv_qmax_alloc;           /* value of qmax used when allocating memory   */
    int cv_indx_acor;            /* index of the zn vector with saved acor      */

    /*-------------------------
      Stability Limit Detection
      -------------------------*/

    bool cv_sldeton;            /* is Stability Limit Detection on?             */
    Real cv_ssdat[6][4];        /* scaled data array for STALD                  */
    int cv_nscon;               /* counter for STALD method                     */
    long int cv_nor;            /* counter for number of order reductions       */

    /*---------------------
      Nonlinear Solver Data
      ---------------------*/

    CorrectorConvergence convfail;            /* flag to indicate when a Jacbian update may
                                                 be needed */

public:

    Vode(MultiStepMethod multistep_method=MultiStepMethod::BDF) {
        cv_lmm = multistep_method;

        // !!! get machine epsilon
        cv_uround = zero;
        
        /* Set default values for integrator optional inputs */
        cv_user_data  = nullptr;
        cv_itol       = ToleranceType::NoneNone;
        cv_qmax = (multistep_method == MultiStepMethod::Adams) ? ADAMS_Q_MAX : BDF_Q_MAX;
        cv_mxstep     = MXSTEP_DEFAULT;
        cv_mxhnil     = MXHNIL_DEFAULT;
        cv_sldeton    = false;
        cv_hin        = zero;
        cv_hmin       = HMIN_DEFAULT;
        cv_hmax_inv   = HMAX_INV_DEFAULT;
        cv_maxnef     = MXNEF;
        cv_maxncf     = MXNCF;
        cv_nlscoef    = CORTES;
        convfail      = CorrectorConvergence::NoFailures;

        /* Set the saved value qmax_alloc */
        cv_qmax_alloc = cv_qmax;
    }

    ~Vode() {}

    void initialize(Real start_time, RealVector<N>& start_vector,
                    Real end_time, VodeStepping vode_step_mode) {

        error_state = InternalErrorState::None;
        warning_state = InternalWarningState::None;
        
        cv_tn = start_time;
        cv_zn[0] = start_vector;
        target_time = end_time;

        assert(vode_step_mode == VodeStepping::Normal || vode_step_mode == VodeStepping::OneStep);

        vode_stepping = vode_step_mode;

        /* Set Current State for a new integration */

        current_state = VodeState::PrepareFirstStep;

        /* Set step parameters */

        cv_q      = 1;
        cv_L      = 2;
        cv_qwait  = cv_L;
        cv_etamax = ETAMX1;

        cv_qu    = 0;
        cv_hu    = zero;
        cv_tolsf = one;

        /* Initialize all the counters */

        num_steps  = 0;
        cv_nfe     = 0;
        cv_ncfn    = 0;
        cv_netf    = 0;
        cv_nni     = 0;
        cv_nsetups = 0;
        cv_nhnil   = 0;
        cv_nstlp   = 0;
        cv_nscon   = 0;

        /* Initialize other integrator optional outputs */

        cv_h0u      = zero;
        cv_next_h   = zero;
        cv_next_q   = 0;

        /* Initialize Stability Limit Detection data */
        cv_nor = 0;
        for (size_t i = 1; i <= 5; i++)
            for (size_t k = 1; k <= 3; k++) 
                cv_ssdat[i-1][k-1] = zero;

        /* Reset checkpoints */
        multistep_checkpoints.unset_all();
        singlestep_checkpoints.unset_all();
        nonlinear_checkpoints.unset_all();
        newton_checkpoints.unset_all();
    }

    void set_tolerances(Real reltol, Real abstol) {
        assert(reltol > zero);
        assert(abstol > zero);

        /* Copy tolerances into memory */
  
        cv_reltol = reltol;
        cv_Sabstol = abstol;

        cv_itol = ToleranceType::ScalarScalar;
    }

    void set_tolerances(Real reltol, RealVector<N>& abstol) {
        assert(reltol > zero);
        assert(abstol.min() > zero);

        /* Copy tolerances into memory */
  
        cv_reltol = reltol;
        cv_Vabstol = abstol;

        cv_itol = ToleranceType::ScalarVector;
    }

    void set_error_weights(RealVector<N>& weights) {
        if (cv_itol == ToleranceType::ScalarScalar) {
            weights = cv_zn[0];
            (weights.abs() * cv_reltol + cv_Sabstol).inv();
        } else if (cv_itol == ToleranceType::ScalarVector) {
            weights = cv_zn[0];
            (weights.abs() * cv_reltol + cv_Vabstol).inv();
        }
    }

    void evaluate_rhs(RealVector<N>& yeval_rhs, Real teval, RealVector<N>& yeval) {
        evaluate_user_rhs(yeval_rhs, teval, yeval, cv_user_data);
        cv_nfe++;
    }

    void initialize_nordsieck_derivative() {
        evaluate_rhs(cv_zn[1], cv_tn, cv_zn[0]);
    }

    void update_state(VodeState new_state) {
        previous_state = current_state;
        current_state = new_state;
    }

    bool assert_error(bool test) {
        if (!test) {
            update_state(VodeState::InternalError);
            return true;
        } else
            return false;
    }

    bool successful() {
        return (error_state == InternalErrorState::None);
    }

    Real get_starting_timestep_upper_bound(Real tdist) {
        RealVector<N> scratch;
        set_error_weights(scratch);
        scratch.inv();

        for (size_t i = 0; i < N; i++) {
            scratch[i] = fabs(cv_zn[1][i])/(HUB_FACTOR * fabs(cv_zn[0][i]) + scratch[i]);
        }

        Real hub_inv = scratch.max_norm();
        Real hub = HUB_FACTOR * tdist;

        if (hub * hub_inv > one) hub = one/hub_inv;
        return hub;
    }

    void set_starting_timestep_bounds() {
        Real tdiff = target_time - cv_tn;

        if (assert_error(tdiff != zero)) {
            error_state = InternalErrorState::TargetIntervalTooSmall;
            return;
        }
        
        int sign = (tdiff > zero) ? 1 : -1;
        Real tdist = fabs(tdiff);
        Real tround = cv_uround * max(fabs(cv_tn), fabs(target_time));

        if (assert_error(tdist >= two*tround)) {
            error_state = InternalErrorState::TargetIntervalTooSmall;
            return;
        }
  
        /* 
           Set lower and upper bounds on h0, and take geometric mean 
           as first trial value.
           Exit with this value if the bounds cross each other.
        */

        start_h_lower_bound = HLB_FACTOR * tround;
        start_h_upper_bound = get_starting_timestep_upper_bound(tdist);
    }

    void apply_starting_timestep_estimate() {
        cv_h = H_BIAS * esthin_h_previous;
        if (cv_h < start_h_lower_bound) cv_h = start_h_lower_bound;
        if (cv_h > start_h_upper_bound) cv_h = start_h_upper_bound;
        if (target_time - cv_tn <= zero) cv_h = -cv_h;
    }

    void evaluate_rhs_starting_timestep() {
        int sign = (target_time - cv_tn > zero) ? 1 : -1;        
        Real time_rhs = cv_tn + esthin_h_previous * sign;
        evaluate_rhs(cv_ydot, time_rhs, cv_ytmp);
        update_state(VodeState::EstimateStartingTimestep);
    }
    
    void estimate_starting_timestep() {
        if (previous_state == VodeState::PrepareFirstStep) {
            set_starting_timestep_bounds();
            if (!successful()) return;
        }

        int sign = (target_time - cv_tn > zero) ? 1 : -1;        
        Real hg  = sqrtf(start_h_lower_bound * start_h_upper_bound);

        if (start_h_upper_bound < start_h_lower_bound) {
            if (sign == -1) cv_h = -hg;
            else            cv_h =  hg;
            update_state(VodeState::ApplyStartingTimestep);
            return;
        } else {
            if (previous_state == VodeState::PrepareFirstStep) {
                esthin_h_previous = hg;
                state_resume_counter = 1;
            }

            if (state_resume_counter < MAX_ITERS) {
                if (previous_state != VodeState::EstimateStartingTimestep_RHS) {
                    // Prepare to evaluate RHS
                    cv_ytmp = cv_zn[1];
                    cv_ytmp * hg * sign + cv_zn[0];

                    // Evaluate RHS
                    update_state(VodeState::EstimateStartingTimestep_RHS);
                    return;
                } else {
                    // RHS is evaluated, continue
                    cv_ydot * one/(hg * sign);
                    cv_ytmp = cv_zn[1];
                    cv_ytmp * (-one/(hg * sign));
                    cv_ydot + cv_ytmp;
                    Real yddnrm = cv_ydot.wrms_norm(cv_ewt);
                    
                    /* Propose new step size */
                    Real hnew = (yddnrm*start_h_upper_bound*start_h_upper_bound > two) ?
                        sqrtf(two/yddnrm) : sqrtf(esthin_h_previous*start_h_upper_bound);
                    Real hrat = hnew/esthin_h_previous;

                    bool found_good_hin = false;

                    if ((hrat > half) && (hrat < two)) {
                        /* Accept hnew if it does not differ from hg by more than a factor of 2 */
                        esthin_h_previous = hnew;
                        found_good_hin = true;
                    } else if ((state_resume_counter > 1) && (hrat > two)) {
                        /* After one pass, if ydd seems to be bad, use fall-back value. */
                        found_good_hin = true;
                    }

                    if (found_good_hin) {
                        apply_starting_timestep_estimate();
                        update_state(VodeState::ApplyStartingTimestep);
                        return;
                    } else {
                        /* Send this value back through f() */
                        esthin_h_previous = hnew;
                        state_resume_counter++;
                        update_state(VodeState::EstimateStartingTimestep);
                        return;
                    }
                }
            } else {
                esthin_h_previous = hg;
                apply_starting_timestep_estimate();
                update_state(VodeState::ApplyStartingTimestep);
                return;
            }
        }
    }

    void evaluate_interpolant(Real time, int derivative_order, RealVector<N>& result) {
        if (assert_error(derivative_order >= 0 && derivative_order <= cv_q)) {
            error_state = InternalErrorState::InvalidInterpolantDerivativeOrderRequested;
            return;
        }

        Real tfuzz =  FUZZ_FACTOR * cv_uround * (fabs(cv_tn) + fabs(cv_hu));

        if (cv_hu < zero) tfuzz = -tfuzz;
        Real tp = cv_tn - cv_hu - tfuzz;
        Real tn1 = cv_tn + tfuzz;

        if (assert_error((time-tp)*(time-tn1) <= zero)) {
            error_state = InternalErrorState::InvalidInterpolantTime;
            return;
        }

        /* Sum the differentiated interpolating polynomial */
        size_t ivec = 0;
        size_t nvec = static_cast<size_t>(cv_q - derivative_order + 1);

        RealVector<Q_MAX+1> cvals;
        RealVector<N> Xvecs[nvec];

        Real s = (time - cv_tn) / cv_h;
        for (int j=cv_q; j >= derivative_order; j--) {
            cvals[ivec] = one;
            for (int i=j; i >= j-derivative_order+1; i--)
                cvals[ivec] *= i;
            for (int i=0; i < j-derivative_order; i++)
                cvals[ivec] *= s;
            Xvecs[ivec] = cv_zn[j];
            ivec++;
        }

        result.sv_linear_combination(nvec, cvals.dataPtr(), Xvecs);
        
        if (derivative_order != 0) {
            result * powf(cv_h,-derivative_order);
        }
    }

    void check_stop_conditions() {
        time_roundoff = FUZZ_FACTOR * cv_uround * (fabs(cv_tn) + fabs(cv_h));
        if (vode_stepping == VodeStepping::Normal) {
            if ((cv_tn - target_time) * cv_h >= zero) {
                cv_tretlast = target_time;
                evaluate_interpolant(target_time, 0, cv_ytmp);
                if (!assert_error(successful()))
                    update_state(VodeState::IntegrationComplete);
                
            }
        } else if (vode_stepping == VodeStepping::OneStep) {
            if (fabs(cv_tn - cv_tretlast) > time_roundoff) {
                post_current_solution();
                update_state(VodeState::IntegrationComplete);
            }
        }
    }

    void apply_starting_timestep() {
        Real timestep_limit_factor = fabs(cv_h) * cv_hmax_inv;
        if (timestep_limit_factor > one) cv_h /= timestep_limit_factor;
        if (fabs(cv_h) < cv_hmin) cv_h *= cv_hmin / fabs(cv_h);

        cv_hscale = cv_h;
        cv_h0u = cv_h;
        cv_hprime = cv_h;
        cv_zn[1] * cv_h;

        update_state(VodeState::AdvanceMultistep);
        
        check_stop_conditions();
    }

    void prepare_first_step() {
        cv_tretlast = cv_tn;
        if (assert_error(cv_itol == ToleranceType::ScalarScalar ||
                         cv_itol == ToleranceType::ScalarVector)) {
            error_state = InternalErrorState::InvalidToleranceType;
            return;
        }
        set_error_weights(cv_ewt);
        cv_jcur = false;
        
        initialize_nordsieck_derivative();
        
        cv_h = cv_hin;
        if (cv_h != zero) {
            if (assert_error((target_time - cv_tn) * cv_h >= zero)) {
                error_state = InternalErrorState::InvalidTimestepSign;
                return;
            } else
                apply_starting_timestep();
        } else {
            update_state(VodeState::EstimateStartingTimestep);
            estimate_starting_timestep();
        }
    }

    void post_current_solution() {
        cv_tretlast = cv_tn;
        cv_ytmp = cv_zn[0];
    }

    void singlestep_prepare() {
        cv_next_h = cv_h;
        cv_next_q = cv_q;

        if (num_steps > 0) set_error_weights(cv_ewt);

        if (assert_error((cv_mxstep > 0) && (num_steps < cv_mxstep))) {
            error_state = InternalErrorState::ExceededMaximumSteps;
            post_current_solution();
            return;
        }

        Real error_norm = cv_zn[0].wrms_norm(cv_ewt);
        cv_tolsf = cv_uround * error_norm;

        if (assert_error(cv_tolsf <= one)) {
            error_state = InternalErrorState::RequestedTooMuchAccuracy;
            post_current_solution();
            cv_tolsf *= two;
        } else {
            cv_tolsf = one;
        }

        if (cv_tn + cv_h == cv_tn) {
            cv_nhnil++;
            if (assert_error(cv_nhnil <= cv_mxhnil)) {
                error_state = InternalErrorState::ExceededMaximumNilSteps;
            } else {
                warning_state = InternalWarningState::UsingNilStepsize;
            }
        }
    }

    void multistep_advance() {
        if (multistep_checkpoints.not_done(MultiStepCheckpoint::Prepare)) {
            singlestep_prepare();
            if (!successful()) return;
            else multistep_checkpoints.set(MultiStepCheckpoint::Prepare);
        }

        if (multistep_checkpoints.not_done(MultiStepCheckpoint::Execute)) {
            singlestep_advance(); // will set Execute as done
        }

        if (multistep_checkpoints.is_done(MultiStepCheckpoint::Execute)) {
            if (!successful()) {
                post_current_solution();
                return;
            }

            num_steps++;

            if ((vode_step_mode == VodeStepping::Normal) && (cv_tn - target_time) * cv_h >= zero) {
                update_state(VodeState::IntegrationComplete);
                cv_tretlast = target_time;
                evaluate_interpolant(target_time, 0, cv_ytmp);
                cv_next_q = cv_qprime;
                cv_next_h = cv_hprime;
            } else if (vode_step_mode == VodeStepping::OneStep) {
                update_state(VodeState::IntegrationComplete);
                post_current_solution();
                cv_next_q = cv_qprime;
                cv_next_h = cv_hprime;
            }

            // unset all checkpoints to prepare for next iteration
            multistep_checkpoints.unset_all();
            singlestep_checkpoints.unset_all();
            nonlinear_checkpoints.unset_all();
            newton_checkpoints.unset_all();
        }
    }

    void predict_solution() {
        /*
         * This routine advances tn by the tentative step size h, and computes
         * the predicted array z_n(0), which is overwritten on zn.  The
         * prediction of zn is done by repeated additions.
         */
        
        cv_tn += cv_h;
        for (size_t k = 1; k <= cv_q; k++)
            for (size_t j = cv_q; j >= k; j--)
                cv_zn[j-1] + cv_zn[j];
    }

    Real alternating_sum(int iend, RealVector<L_MAX>& a, int k) {
        /*  
         * cvAltSum returns the value of the alternating sum
         *   sum (i= 0 ... iend) [ (-1)^i * (a[i] / (i + k)) ].
         * If iend < 0 then cvAltSum returns 0.
         * This operation is needed to compute the integral, from -1 to 0,
         * of a polynomial x^(k-1) M(x) given the coefficients of M(x).
         */

        if (iend < 0) return(zero);
  
        Real sum = zero;
        int sign = 1;

        for (size_t i=0; i <= iend; i++) {
            sum += sign * (a[i] / (i+k));
            sign = -sign;
        }
        
        return(sum);        
    }

    void generate_adams_coefficients(RealVector<L_MAX>& m, Real& hsum) {
        /*
         * This routine generates in m[] the coefficients of the product
         * polynomial needed for the Adams l and tq coefficients for q > 1.
         */

        Real xi_inv, sum;
  
        hsum = cv_h;
        m[0] = one;
        for (size_t i=1; i <= cv_q; i++) m[i] = zero;
        for (size_t j=1; j < cv_q; j++) {
            if ((j==cv_q-1) && (cv_qwait == 1)) {
                sum = alternating_sum(cv_q-2, m, 2);
                cv_tq[1] = cv_q * sum / m[cv_q-2];
            }
            xi_inv = cv_h / hsum;
            for (size_t i=j; i >= 1; i--) m[i] += m[i-1] * xi_inv;
            hsum += cv_tau[j];
            /* The m[i] are coefficients of product(1 to j) (1 + x/xi_i) */
        }
    }

    void finish_adams_update(RealVector<L_MAX>& m, RealVector<3>& M, Real hsum) {
        /*
         * This routine completes the calculation of the Adams l and tq.
         */
        
        Real M0_inv, xi, xi_inv;
  
        M0_inv = one / M[0];
  
        cv_l[0] = one;
        for (size_t i=1; i <= cv_q; i++)
            cv_l[i] = M0_inv * (m[i-1] / static_cast<Real>(i));
        xi = hsum / cv_h;
        xi_inv = one / xi;
  
        cv_tq[2] = M[1] * M0_inv / xi;
        cv_tq[5] = xi / cv_l[cv_q];

        if (cv_qwait == 1) {
            for (size_t i=cv_q; i >= 1; i--) m[i] += m[i-1] * xi_inv;
            M[2] = alternating_sum(cv_q, m, 2);
            cv_tq[3] = M[2] * M0_inv / cv_L;
        }

        cv_tq[4] = cv_nlscoef / cv_tq[2];
    }

    void update_method_adams() {
        /*
         * This routine handles the computation of l and tq for the
         * case lmm == CV_ADAMS.
         *
         * The components of the array l are the coefficients of a
         * polynomial Lambda(x) = l_0 + l_1 x + ... + l_q x^q, given by
         *                          q-1
         * (d/dx) Lambda(x) = c * PRODUCT (1 + x / xi_i) , where
         *                          i=1
         *  Lambda(-1) = 0, Lambda(0) = 1, and c is a normalization factor.
         * Here xi_i = [t_n - t_(n-i)] / h.
         *
         * The array tq is set to test quantities used in the convergence
         * test, the error test, and the selection of h at a new order.
         */

        RealVector<L_MAX> m;
        RealVector<3> M;
        Real hsum;
  
        if (cv_q == 1) {
            cv_l[0] = cv_l[1] = cv_tq[1] = cv_tq[5] = one;
            cv_tq[2] = half;
            cv_tq[3] = one/twelve;
            cv_tq[4] = cv_nlscoef / cv_tq[2];       /* = 0.1 / tq[2] */
            return;
        }

        generate_adams_coefficients(m, hsum);
  
        M[0] = alternating_sum(cv_q-1, m, 1);        
        M[1] = alternating_sum(cv_q-1, m, 2);
  
        finish_adams_update(m, M, hsum);        
    }

    void update_method_bdf() {
        // !! Don
    }

    void update_method() {
        /*
         * This routine is a high level routine which calls cvSetAdams or
         * cvSetBDF to set the polynomial l, the test quantity array tq, 
         * and the related variables  rl1, gamma, and gamrat.
         *
         * The array tq is loaded with constants used in the control of estimated
         * local errors and in the nonlinear convergence test.  Specifically, while
         * running at order q, the components of tq are as follows:
         *   tq[1] = a coefficient used to get the est. local error at order q-1
         *   tq[2] = a coefficient used to get the est. local error at order q
         *   tq[3] = a coefficient used to get the est. local error at order q+1
         *   tq[4] = constant used in nonlinear iteration convergence test
         *   tq[5] = coefficient used to get the order q+2 derivative vector used in
         *           the est. local error at order q+1
         */

        if (cv_lmm == MultiStepMethod::Adams) {
            update_method_adams();
        } else if (cv_lmm == MultiStepMethod::BDF) {
            update_method_bdf();
        }

        cv_rl1 = one / cv_l[1];
        cv_gamma = cv_h * cv_rl1;
        if (num_steps == 0) cv_gammap = cv_gamma;
        cv_gamrat = (num_steps > 0) ? cv_gamma / cv_gammap : one;
    }

    void singlestep_advance() {
        if (singlestep_checkpoints.not_done(SingleStepCheckpoint::Prepare)) {
            saved_time = cv_tn;
            number_step_corrector_failures = 0;
            number_step_error_failures = 0;

            if ((num_steps > 0) && (cv_hprime != cv_h)) {
                // cvAdjustParams
            }
            
            singlestep_checkpoints.set(SingleStepCheckpoint::Prepare);
        }

        if (singlestep_checkpoints.not_done(SingleStepCheckpoint::PredictAndSet) &&
            singlestep_checkpoints.done(SingleStepCheckpoint::Prepare)) {
            predict_solution();
            update_method();
            singlestep_checkpoints.set(SingleStepCheckpoint::PredictAndSet);
        }

        if (singlestep_checkpoints.not_done(SingleStepCheckpoint::SolveNonlinear) &&
            singlestep_checkpoints.done(SingleStepCheckpoint::PredictAndSet)) {
            // cvNls : will set SolveNonLinear as done
        }

        if (singlestep_checkpoints.not_done(SingleStepCheckpoint::EvaluateNonlinear) &&
            singlestep_checkpoints.done(SingleStepCheckpoint::SolveNonlinear)) {
            // ...
            // unset PredictAndSet, SolveNonlinear, and EvaluateNonlinear to loop again
        }

        if (singlestep_checkpoints.not_done(SingleStepCheckpoint::FinishStep) &&
            singlestep_checkpoints.done(SingleStepCheckpoint::EvaluateNonlinear)) {

        }
        
    }

    void advance() {
        if (current_state == VodeState::PrepareFirstStep)
            prepare_first_step();
        else if (current_state == VodeState::EstimateStartingTimestep)
            estimate_starting_timestep();
        else if (current_state == VodeState::EstimateStartingTimestep_RHS)
            evaluate_rhs_starting_timestep();
        else if (current_state == VodeState::ApplyStartingTimestep)
            apply_starting_timestep();
        else if (current_state == VodeState::AdvanceMultistep)
            multistep_advance();
    }

};

#endif
