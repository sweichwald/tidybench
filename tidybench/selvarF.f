c Implements the SELVAR (Selective auto-regressive model) algorithm.
c
c Based on an implementation that is originally due to Gherardo Varando (gherardovarando).
c
      SUBROUTINE SLVAR(T, N, X, BS, ML, MXITR, B, A, INFO, TRC)  
c     Select structure and lags of a (local) VAR model
c     Gherardo Varando (2019)
      INTEGER T, N, BS, ML, MXITR, A(N,N), INFO, TRC
      DOUBLE PRECISION X(T,N), B(N,N)
Cf2py integer, intent(in) n 
Cf2py integerm, intent(in) t
Cf2py intent(in) x
Cf2py integer, optional, intent(in):: bs = -1
Cf2py integer, optional, intent(in) :: ml = -1 
Cf2py integer, optional, intent(in) :: mxitr = -1
Cf2py optional, intent(in) :: lmb = 0
Cf2py intent(out) b
Cf2py intent(out) a 
Cf2py intent(out)  info
Cf2py integer, optional, intent(in) :: trc = 1
c     ON ENTRY
c        T integer
c          number of time points
c        N integer
c          number of variables 
c        X double precision (T, N)
c          matrix of observations
c        BS integer
c           size of the batch 
c            * IF BS .GT. 0 batch size = BS
c            * IF BS .LT. 0 batch size = T / -BS 
c            * IF BS .EQ. 0 
c        ML integer
c           maximum time lag 
c            * IF ML .GT. 0 maximum time lag = ML 
c            * IF ML .LT. 0 maximum time lag is searched iteratively  
c        MXITR integer 
c           maximum number of iterations in hc
c            * IF MXITR .EQ. 0 perform no search and just score edges
c            * IF MXITR .LT. 0 THEN MXITR = INF
c        B double precision (N,N) 
c          empty matrix
c        A integer (N,N) 
c          empty matrix for return 
c        TRC integer
c            IF TRC .GT. 0 print trace information
c     ON RETURN
c        B double precision (N, N)
c          matrix of scores
c        A integer (N,N)
c          matrix of estimated lags 
c        INFO integer 
c          information on errors
c     DETAILS 
c       
c     SUBROUTINES 
c       * GTPRSS 
c       * GTCOEF
c       * GTRSS 
c       * DGELS from LAPACK
c       * DORGQR from LAPACK
c     INTERNAL VARIABLES
      INTEGER I, J, K, FLG, ITR, TMP, IBST, KBST, ITML
      DOUBLE PRECISION XX(T, N + 1), YY(T, 1), 
     *                 WK(2*T*N), 
     *                 SCR, NWSCR, TMPSCR
      SCR = 0.0
      NWSCR = 0.0
      FLG = 0 
      ITR = 0
      ITML = 0
      IF (ML .LT. 1) ITML = 1
      IF (ML .GE. T .OR. ML .LT. 1) ML = 1
      IF (BS .LT. 0) BS = (T - ML) / (-BS) 
      IF (BS .GT. T - ML) BS = T - ML
      IF (MXITR .EQ. 0) GOTO 100 
c     print parameters
      IF (TRC .GT. 10) THEN
         WRITE(*,*) BS, ML 
      ENDIF
      DO 20 J = 1,N
         DO 10 I = 1,N
c        initialize the empty graph 
            A(I,J) = 0
 10      CONTINUE
 20   CONTINUE           
      DO 50 J=1,N 
         ITR = 0
         IF (ITML .GT. 0) ML = 1
c        compute initial score (prss) for j
         CALL GTPRSS(T, N, X, ML, BS, A, J, XX, YY, WK, 
     *                SCR, INFO)
c        hill-climb search start for j 
 500     CONTINUE
c        increase iteration counter
         ITR = ITR + 1
         FLG = 0
         TMPSCR = SCR
         IBST = -1
         DO 40 K= 0,ML  
            DO 30 I=1,N 
               TMP = A(I,J)  
               IF (K .NE. TMP) THEN
                 A(I,J) = K 
                 CALL GTPRSS(T, N, X, ML, BS, A, J, XX, YY, WK,
     *                     NWSCR, INFO)
                 IF (NWSCR .GE. 0 .AND. NWSCR .LT. TMPSCR) THEN
                    TMPSCR = NWSCR 
                    IBST = I
                    KBST = K
                 ENDIF
                 A(I,J) = TMP
              ENDIF
 30        CONTINUE    
 40      CONTINUE           
         IF (IBST .GT. 0) THEN
            A(IBST, J) = KBST
            FLG = 1
            SCR = TMPSCR
            IF (TRC .GT. 0) THEN 
               WRITE(*,"(a,a5,i3,a5,i3,a,i3,a5,i3)", ADVANCE = "NO") 
     *                                    char(13), "ITER:", ITR, 
     *                                             " ADD ", IBST,"-",J, 
     *                                             " LAG=", KBST 
            ENDIF
         ENDIF     
         IF (ITML .GT. 0) ML = MIN(ML + 1, T / 2)
      IF ((MXITR .LT. 0 .OR. ITR .LT. MXITR) .AND. FLG .GT. 0) GOTO 500
 50   CONTINUE        
 100  CONTINUE
      CALL GTCOEF(T, N, X, ML, BS, A, "ABS", 0, XX, YY, WK, B,
     *             INFO)
      MXITR = ITR
      RETURN
c     last line of SLVAR
      END
c
c      
      SUBROUTINE GTPRSS(T, N, X, ML, BS, A, J, XX, YY, WK, SCR,
     *                 INFO)
c     Get average Predicted RSS for a given variable  
c     Gherardo Varando (2019)
      INTEGER T, N, J, ML, A(N,N), INFO, BS
      DOUBLE PRECISION X(T,N), XX(T, N + 1), YY(T, 1), 
     *                 WK(2*T*N), SCR
Cf2py intent(in) n 
Cf2py intent(in) t
Cf2py intent(in) x
Cf2py intent(in) bs 
Cf2py intent(in)  ml 
Cf2py intent(in)  a 
Cf2py intent(in)  j 
Cf2py optional, intent(cache) ::  xx = array((t,n+1))
Cf2py optional, intent(cache) ::  yy = array((t,1))
Cf2py optional, intent(cache) ::  wk = array(2*t*n)
Cf2py intent(out)  scr
Cf2py intent(out)  info
c  
c     ON ENTRY 
c        T, N, ML, BS, X, A  as in DSELVAR  
c        XX, YY, WK  working variables  
c     ON RETURN
c        SCR the computed average PRSS 
c        INFO 
c     INTERNAL VARIABLES 
      INTEGER NF, NV, TT, I, K 
      DOUBLE PRECISION TMP, TMPY
      IF (ML .GE. T .OR. ML .LT. 1) ML = 1
      IF (BS .LT. 0) BS = (T - ML) / (-BS) 
      IF (BS .GT. T - ML) BS = T - ML
      SCR = 0.0
      NF = (T - ML) / BS
      DO 100 K = 1, NF
         NV = 1
         DO 5  TT = 1, BS
            XX(TT, NV) = 1   
            YY(TT, 1) = X(TT + ML + (K-1) * BS, J) 
 5       CONTINUE
         DO 20 I = 1, N 
            IF (A(I,J) .GT. 0) THEN 
               NV = NV + 1
               IF (NV .GT. BS) THEN
                  SCR = -1
                  GOTO 110
               ENDIF
               DO 10 TT = 1, BS
                  XX(TT, NV) = X(TT + ML - A(I,J) + (K - 1)*BS, I)  
 10            CONTINUE
            ENDIF
 20      CONTINUE
         CALL DGELS("N", BS, NV, 1, XX, T,
     *               YY, T, WK, 2*T*N, INFO)
         IF (INFO .NE. 0) GOTO 110
c        compute predictive sum of squares,
         CALL DORGQR(NV, NV, NV, XX, T, WK(1), 
     *               WK(NV + 1), 2*T*N - NV, INFO) 
         DO 80 TT = 1, BS
            TMPY = X(TT + ML + (K-1)*BS, J) - YY(1,1)
            NV = 1
            TMP = XX(TT, 1) ** 2 
            DO 70 I = 1,N
               IF (A(I,J) .GT. 0) THEN 
                  NV = NV + 1
                  TMP = TMP + (XX(TT,NV) ** 2)
                  TMPY = TMPY - 
     *                   (X(TT + ML - A(I,J) + (K-1)*BS, I) * YY(NV,1)) 
               ENDIF
 70         CONTINUE
            SCR = SCR + (((TMPY) / (1 - TMP)) ** 2) 
 80      CONTINUE              
 100  CONTINUE        
 110  CONTINUE
      RETURN
      END
c
c  
      SUBROUTINE GTCOEF(T, N, X, ML, BS, A,JOB,NRM, XX, YY, WK, B,
     *                 INFO)
c     Get average (and sqared or absolute) coefficients 
c     coefficients can be normalized by 
c        b(i,J) = b(i,j) / sqrt(b(i,j)^2 + v(j) / v(i))   
c     where b(i,j) is the coefficient of x(,i) in the regression of
c     x(,j) and v(i) is the varaince of the residuals for x(,i) 
c     Gherardo Varando (2019) 
      INTEGER T, N, ML, BS, A(N,N), INFO 
      DOUBLE PRECISION X(T,N), XX(T, N + 1), YY(T, 1), 
     *                 WK(2*T*N), B(N,N)
      CHARACTER JOB*3
Cf2py intent(in) t
Cf2py intent(in) n 
Cf2py intent(in) ml 
Cf2py intent(in) bs 
Cf2py intent(in) x
Cf2py intent(in) a 
Cf2py optional, intent(in) job
Cf2py optional, intent(in) :: nrm = 0
Cf2py optional, intent(cache) :: xx=array((t,n+1))
Cf2py optional, intent(cache) :: yy=array((t,n+1))
Cf2py optional, intent(cache) :: wk=array((2*t*n))
Cf2py intent(out)  b
Cf2py intent(out)  info
c          
c     ON ENTRY 
c        T, N, ML, BS, X, A  as in SLVAR  
c        JOB character 
c            IF JOB .EQ. "ABS" the average absolute coefficients
c            IF JOB .EQ. "SQR" the average square coefficients 
c            ELSE the average coefficients 
c        NRM integer
c            IF NRM .GT. 0 normalize the coefficient
c        XX, YY, WK  working variables  
c     ON RETURN
c        B the computed average coefficients  
c        INFO 
c     INTERNAL VARIABLES 
      INTEGER NF, NV, TT, I, K, J
      DOUBLE PRECISION V(N)
      IF (ML .GE. T .OR. ML .LT. 1) ML = 1
      IF (BS .LT. 0) BS = (T - ML) / (-BS) 
      IF (BS .GT. T - ML) BS = T - ML
      NF = (T - ML) / BS
      DO 200 J = 1, N
         V(J) = 0
         DO 1 I = 1,N
            B(I,J) = 0
 1       CONTINUE
         DO 100 K = 1, NF
            NV = 1
            DO 5  TT = 1, BS
               XX(TT, NV) = 1   
               YY(TT, 1) = X(TT + ML + (K-1) * BS, J) 
 5          CONTINUE
            DO 20 I = 1, N 
               IF (A(I,J) .GT. 0) THEN 
                  NV = NV + 1
                  DO 10 TT = 1, BS
                     XX(TT, NV) = X(TT + ML - A(I,J) + (K - 1)*BS, I)  
 10               CONTINUE
               ENDIF
 20         CONTINUE
            CALL DGELS("N", BS, NV, 1, XX, T,
     *                 YY, T, WK, 2*T*N, INFO)
            IF (INFO .NE. 0) GOTO 100
            DO 30 I=NV+1, BS
               V(J) = V(J) + YY(I,1) ** 2 / (BS * NF)
 30         CONTINUE
            NV = 1
            DO 40 I = 1, N 
               IF (A(I,J) .GT. 0) THEN 
                  NV = NV + 1
                  IF (JOB .EQ. "ABS") THEN
                     B(I,J) = B(I,J) + (ABS(YY(NV,1))  / NF)
                  ELSEIF (JOB .EQ. "SQR") THEN
                     B(I,J) = B(I,J) + (YY(NV,1)**2)/NF
                  ELSE
                     B(I,J) = B(I,J) + YY(NV,1)/NF 
                  ENDIF
               ENDIF
 40         CONTINUE
 100     CONTINUE        
 200  CONTINUE
      IF (NRM .GT. 0) THEN 
         DO 300 J = 1,N
            DO 250 I = 1,N
               B(I,J) = B(I,J) / SQRT( B(I,J)**2 + V(J)/V(I))
 250        CONTINUE
 300     CONTINUE
      ENDIF
      RETURN
      END
c 
c
      SUBROUTINE GTRSS(T, N, X, ML, BS, A, J, XX, YY, WK, SCR,
     *                 INFO)
c     get average residuals sum of squares for variable j  
c     Gherardo Varando (2019)
      INTEGER T, N, J, ML, A(N,N), INFO, BS
      DOUBLE PRECISION X(T,N), XX(T, N + 1), YY(T, 1), 
     *                 WK(2*T*N),   SCR
c          
Cf2py intent(in) t
Cf2py intent(in) n 
Cf2py intent(in) ml 
Cf2py intent(in) bs 
Cf2py intent(in) x
Cf2py intent(in) a 
Cf2py intent(in) j 
Cf2py optional, intent(cache) :: xx=array((t,n+1))
Cf2py optional, intent(cache) :: yy = array((t,1))
Cf2py optional, intent(cache) :: wk = array(2*t*n)
Cf2py intent(out)  scr
Cf2py intent(out)  info
c          
c     ON ENTRY 
c        T, N, ML, BS, X, A  as in SLVAR  
c        XX, YY, WK  working variables  
c        J INTEGER  the variable to consider
c     ON RETURN
c        SCR the computed RSS for variable J 
c        INFO 
c     INTERNAL VARIABLES 
      INTEGER NF, NV, TT, I, K 
      IF (ML .GE. T .OR. ML .LT. 1) ML = 1
      IF (BS .LT. 0) BS = (T - ML) / (-BS) 
      IF (BS .GT. T - ML) BS = T - ML
      SCR = 0.0
      NF = (T - ML) / BS
      DO 100 K = 1, NF
         NV = 1
         DO 5  TT = 1, BS
            XX(TT, NV) = 1   
            YY(TT, 1) = X(TT + ML + (K-1) * BS, J) 
 5       CONTINUE
         DO 20 I = 1, N 
            IF (A(I,J) .GT. 0) THEN 
               NV = NV + 1
               DO 10 TT = 1, BS
                  XX(TT, NV) = X(TT + ML - A(I,J) + (K - 1)*BS, I)  
 10            CONTINUE
            ENDIF
 20      CONTINUE
         CALL DGELS("N", BS, NV, 1, XX, T,
     *               YY, T, WK, 2*T*N, INFO)
         IF (INFO .NE. 0) GOTO 100
c        compute RSS,
         DO 30 TT = NV+1, BS
           SCR = SCR + (YY(TT,1) ** 2) 
 30      CONTINUE
         SCR = SCR
 100  CONTINUE        
      SCR = SCR / (NF * BS)
      RETURN
      END
c
c
c
      SUBROUTINE GTSTAT(T, N, X, BS, ML, A, JOB, XX, YY, WK, B, DF)  
c     Obtain the log likelihood-ratio statistics, the f-statistics 
c     or the difference of residuals for each edge
c     Gherardo Varando (2019)
      INTEGER T, N, BS, ML, INFO, A(N,N), DF(N, 2)
      DOUBLE PRECISION X(T,N), XX(T,N + 1), YY(T,1),WK(2*T*N), B(N,N) 
      CHARACTER JOB*2
Cf2py intent(in) t
Cf2py intent(in) n 
Cf2py intent(in) ml 
Cf2py intent(in) bs 
Cf2py intent(in) x
Cf2py intent(in) a 
Cf2py intent(in) job
Cf2py optional, intent(cache) ::  xx = array((t,n))
Cf2py optional, intent(cache) ::  yy = array((t,1))
Cf2py optional, intent(cache) ::  wk = array(2*t*n) 
Cf2py intent(out)  b
Cf2py intent(out)  df
c          
c     ON ENTRY 
c        T, N, ML, BS, X, A  as in SLVAR  
c        XX, YY, WK  working variables  
c        JOB CHARACHTER*2 
c            IF "LR" the logarithm of the likelihood-ratio 
c            IF "FS" the F-statistic 
c            IF "ÐF" the difference of RSS
c     ON RETURN
c        B DOUBLE PRECISION(N,N)
c          the requested statistics
c        DF DOUBLE PRECISION(N,2)
c          values to obtain degrees of freedom 
c     INTERNAL VARIABLES 
      INTEGER I,J,TMP
      DOUBLE PRECISION SCR,NWSCR
      IF (ML .LT. 1) THEN 
         DO 10 J = 1,N 
            DO 5 I = 1,N 
               ML = MAX(ML,A(I,J))  
 5          CONTINUE
 10      CONTINUE
      ENDIF
      IF (ML .GE. T .OR. ML .LT. 1) ML = 1
      IF (BS .LT. 0) BS = (T - ML) / (-BS) 
      IF (BS .GT. T - ML) BS = T - ML      
      NF = (T - ML) / BS
      DO 60 J = 1,N 
         DF(J, 1) = 0
         DF(J, 2) = 0
c        get rss for variable j 
         CALL GTRSS(T, N, X, ML,BS, A, J, XX, YY, WK, 
     *              SCR, INFO)
         DO 55 I=1,N 
            B(I,J) = 0
            IF (A(I,J) .GT. 0) THEN
c              add one parameter for each batch 
               DF(J, 1) = DF(J, 1) + NF
c              remove one edge from matrix a 
               TMP = A(I,J) 
               A(I,J) = 0 
c              compute new score
               CALL GTRSS(T, N, X, ML, BS, A, J, XX, YY, WK, 
     *                       NWSCR, INFO)
c              restore matrix a 
               A(I,J) = TMP    
c              store relevant statistic
               IF (JOB .EQ. "FS") B(I,J) = (NWSCR - SCR) / SCR
               IF (JOB .EQ. "LR") B(I,J) = (LOG(NWSCR) - LOG(SCR)) 
     *                                     * NF * BS
               IF (JOB .EQ. "DF") B(I,J) = NWSCR - SCR 
               ENDIF
 55         CONTINUE
            DF(J,2) = DF(J,1) - NF
 60      CONTINUE
c     for f-statistics, finish computing and store df
      IF (JOB .EQ. "FS") THEN
         DO 70 J = 1,N
            DF(J,2) = BS*NF - DF(J,1) 
            DF(J,1) = NF 
            DO 65 I = 1,N 
               B(I,J) = B(I,J) * DF(J,2)
 65         CONTINUE
 70      CONTINUE              
      ENDIF
      RETURN
      END      
