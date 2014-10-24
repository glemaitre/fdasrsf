#include <iostream>
#include "armadillo"
#include "bayes_funcs.hpp"

using namespace arma;
using namespace std;

struct simu_result {
    vec best_match;
    mat match_collect;
    vec dist_collect; 
    vec kappafamily;
    vec log_posterior;
    float dist_min;
}

void simuiter(int iter, int p, vec qt1_5, vec qt2_5, int L, float tau, int times, float kappa, float alpha, float beta, float powera, float dist, float dist_min, vec best_match, vec match, int thin, int cut){
    
    uvec Ixout(2*times);
    uvec Ioriginal(L+1);
    float increment,n_dist,o_dist,adjustcon,ratio,prob,u,logpost;
    int increment_int,newj,tempnew,tempold,tempx;
    vec newmatchvec(3),oldmatchvec(3),idenmatchvec(3),pnewvec(2),poldvec(2);
    vec interynew(2*times),interyold(2*times),interx(2*times),xout(2*times),internew(2*times),interold(2*times);
    vec qt1_5_interx(2*times),qt2_5_internew(2*times),qt2_5_interold(2*times),diff_ynew(2*times),diff_yold(2*times);
    vec original(L+1),idy(p),scalevec(p),qt_5_fitted(p),kappa_collect(iter),log_collect(iter),dist_collect(iter);
    uvec Irow = seq_len(p);
    vec row = conv_to<vec>::from(Irow);
    rowvec scale(L);
    mat match_collect(iter/thin,L+1);

    int q2LLlen = (p-1)*times+1;
    vec q2LL(q2LLlen);
    uvec tempspan1(q2LLlen);
    tempspan1.fill(0);
    tempspan1 = seq_len(q2LLlen);
    vec temp_span2 = conv_to<vec>::from(tempspan1);
    float timesf = (float)(times);
    vec q2LL_time = temp_span2*(1/timesf);
    uvec q2L_time1 = seq_len(p);
    vec q2L_time2 = conv_to<vec>::from(q2L_time1);
    q2LL = approx(p,q2L_time2,qt2_5,q2LLlen,q2LL_time);
    
    for (int j = 0; j < iter; j++)
    { 
        for (int i = 1; i < L; i++)       
        {
            if ((match(i+1)-match(i-1))>2)
            {  
                vec tmp = randn<vec>(1)*tau; 
                increment = tmp(0);
                increment_int = round(increment);
                if (increment_int == 0) {increment_int = (increment>0)?(1):(-1);}
                newj = match(i) + increment_int;
                if((newj < match[i+1]) && (newj > match[i-1]))    
                {
                    newmatchvec(0) = match(i-1);
                    newmatchvec(1) = newj;
                    newmatchvec(2) = match(i+1);
                    oldmatchvec(0) = match(i-1);
                    oldmatchvec(1) = match(i);
                    oldmatchvec(2) = match(i+1);
                    idenmatchvec(0) = times*(i-1);
                    idenmatchvec(1) = times*i;
                    idenmatchvec(2) = times*(i+1);
                    Ixout = seq_len(2*times)+times*(i-1);
                    xout = conv_to<vec>::from(Ixout);
                    internew = approx(3,idenmatchvec,newmatchvec,2*times,xout);
                    interold = approx(3,idenmatchvec,oldmatchvec,2*times,xout);
                    interx = xout;
                    interynew = internew;
                    interynew.insert_rows(2*times,1);
                    interynew(2*times) = match(i+1);
                    interyold = interold;
                    interyold.insert_rows(2*times,1);
                    interyold(2*times) = match(i+1);
                    diff_ynew = interynew.rows(1,2*times)-interynew.rows(0,2*times-1);
                    diff_yold = interyold.rows(1,2*times)-interyold.rows(0,2*times-1);
                    for (int ll=0;ll< (2*times); ll++)
                    {
                        tempx = round(interx[ll]);
                        qt1_5_interx[ll] = qt1_5[tempx];
                        internew[ll] = (internew[ll] > (p-1))?(p-1):(internew[ll]);
                        tempnew = round(times*internew[ll]);
                        qt2_5_internew[ll] = q2LL(tempnew)*sqrt(diff_ynew[ll]);
                        interold[ll] = (interold[ll] > (p-1))?(p-1):(interold[ll]);
                        tempold = round(times*interold[ll]);
                        qt2_5_interold[ll] = q2LL[tempold]*sqrt(diff_yold[ll]);
                    }
                    n_dist = pow(norm(qt1_5_interx-qt2_5_internew,2),2)/p;
                    o_dist = pow(norm(qt1_5_interx-qt2_5_interold,2),2)/p;
                    pnewvec = (newmatchvec.rows(1,2)-newmatchvec.rows(0,1))/p;
                    poldvec = (oldmatchvec.rows(1,2)-oldmatchvec.rows(0,1))/p;
                    adjustcon = exp((powera-1)*(sum(log(pnewvec))-sum(log(poldvec))));
                    ratio = adjustcon*exp(kappa*o_dist-kappa*n_dist);
                    prob = (ratio < 1)?(ratio):(1);
                    
                    vec temp2 = randu<vec>(1);
                    u = temp2(0);
                    match[i] = (u < prob)?(newj):(match[i]);
                }
            }
        }   

        Ioriginal = (seq_len(L+1))*times;
        original = conv_to<vec>::from(Ioriginal);
        idy = round(approx(L+1,original,match,p,row)); 
        for (int ii = 0;ii<L;ii++){scale[ii] = sqrt((match[ii+1]-match[ii])/times);}
        for (int kk=0;kk<p;kk++)
        {   
            idy[kk] = (idy[kk]<p)?(idy[kk]):(p-1);
            scalevec[kk] = scale[kk/times];
            qt_5_fitted[kk] = scalevec[kk]*qt2_5[idy[kk]];
        }
        dist =  pow(norm(qt1_5-qt_5_fitted,2),2)/p;
        dist_collect[j] = dist; 
        if (dist < dist_min)
        {
            best_match = match;
        dist_min = dist;
        }
        if(j%thin==0){ match_collect.row(j/thin) =  match.t();}
        default_random_engine generator;
        gamma_distribution<float> rgamma(p/2+alpha, 1/(dist+beta));
        kappa = rgamma(generator);
        kappa_collect[j] = kappa;
logpost = (p/2+alpha)*log(kappa)-kappa*(beta+dist);
log_collect[j] = logpost;
}
vec Rr_best_match = best_match+1;
mat Rr_match_collect = match_collect+1;

    
}
