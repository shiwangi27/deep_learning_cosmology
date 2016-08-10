subroutine lca(basis, stimuli, eta, lamb, nIter, softThresh, adapt, s, u, thresh, nBasis, nStimuli, length)
  implicit none
!
! Inputs
!
  integer, parameter :: dp = KIND(1.0d0)
  integer, parameter :: li = SELECTED_INT_KIND(8)
  real(dp), parameter :: alpha = 1.0
  real(dp), parameter :: beta = 0.0
  integer(li), intent(in) :: nIter, softThresh, nBasis, nStimuli, length
  real(dp), intent(in), dimension(0:nBasis-1, 0:length-1) :: basis
  real(dp), intent(in), dimension(0:nStimuli-1, 0:length-1) :: stimuli
  real(dp), intent(in) :: eta, lamb, adapt
!
! Outputs
!
  real(dp), intent(inout), dimension(0:nStimuli-1, 0:nBasis-1) :: u
  !f2py intent(inout) :: u
  real(dp), intent(inout), dimension(0:nStimuli-1, 0:nBasis-1) :: s
  !f2py intent(inout) :: s
  real(dp), intent(inout), dimension(0:nStimuli-1) :: thresh
  !f2py intent(inout) :: thresh

  real(dp), dimension(0:nStimuli-1, 0:nBasis-1) :: b, ci
  real(dp), dimension(0:nBasis-1,0:nBasis-1) :: c
  integer(li) :: ii,jj,kk

  external :: DGEMM, DSYMM
  real(dp), external :: DDOT

  call DGEMM("n","t",nBasis,nBasis,length,alpha,basis,nBasis,basis,nBasis,beta,c,nBasis)
  do ii=0,nbasis-1
     c(ii,ii) = 0.0
  end do
  call DGEMM("n","t",nStimuli,nBasis,length,alpha,stimuli,nStimuli,basis,nBasis,beta,b,nStimuli)
  do ii=0,nStimuli-1
    do jj=0,nBasis-1
      thresh(ii) = thresh(ii)+ABS(b(ii,jj))
    end do
    thresh(ii) = thresh(ii)/nBasis
  end do
  do jj=0,nIter-1
     call DSYMM("r","l",nStimuli,nBasis,alpha,c,nBasis,s,nStimuli,beta,ci,nStimuli)
     u = eta*(b-ci)+(1-eta)*u
     do kk=0,nBasis-1
        do ii=0,nStimuli-1
           if ((u(ii,kk) < thresh(ii)) .and. (u(ii,kk) > -thresh(ii))) then
              s(ii,kk) = 0.
           else if (softThresh .eq. 1) then
              s(ii,kk) = u(ii,kk)-sign(u(ii,kk),u(ii,kk))*thresh(ii)
           else
              s(ii,kk) = u(ii,kk)
           end if
        end do
     end do
     do ii=0,nStimuli-1
        if (thresh(ii) > lamb) then
           thresh(ii) = adapt*thresh(ii)
        end if
     end do
  end do
end subroutine

