#include <iostream>
#include "armadillo"
#include "bayes_funcs.hpp"

using namespace arma;
using namespace std;

struct dp_result {
    uvec J;
    double NDist;
    vec q2LL;
};

dp_result dp_bayes(vec q1, vec q1L, vec q2L, int times, int cut){

    int colnum = q1L.size();
    int rownum = colnum/times;
    int q2LLlen = (colnum-1)*times+1;
    vec q2LL(q2LLlen);
    q2LL.fill(0);
    uvec tempspan1(q2LLlen);
    tempspan1.fill(0);
    tempspan1 = seq_len(q2LLlen);
    vec temp_span2 = conv_to<vec>::from(tempspan1);
    float timesf = (float)(times);
    vec q2LL_time = temp_span2*(1/timesf);
    uvec q2L_time1 = seq_len(colnum);
    vec q2L_time2 = conv_to<vec>::from(q2L_time1);
    q2LL = approx(colnum,q2L_time2,q2L,q2LLlen,q2LL_time);
    mat ID(rownum+1,colnum+1);
    ID.fill(0);
    mat S(rownum+1,colnum+1);
    S.fill(0);
    int jend,start,end,k;
    uword index;
    uvec interx(times-1);
    interx.fill(0);
    vec intery2(times-1);
    intery2.fill(0);
    uvec intery(times-1);
    intery.fill(0);
    uvec span1 = seq_len(times-1)+1;
    vec span2 = conv_to<vec>::from(span1);
    vec q1x(times-1);
    q1x.fill(0);
    vec q2y(times-1);
    q2y.fill(0);

    for (int i = 1;i < rownum ;i++)
    {
        jend = ((i*cut+1)< colnum)?(i*cut+1):(colnum);

        for (int j=i;j< jend;j++)
        {
            uvec tmp(2);
            tmp << i-1 << j-cut;
            start = max(tmp);
            tmp << j-1 << cut*(i-1);
            end   = min(tmp);
            uvec n = start+seq_len(end-start+1);
            k = n.n_elem;
            interx = times*(i-1)+span1;
            vec Energy(k);

            for(int m = 0; m < k ; m++)
            {
                float jf = j;
                intery2 = ((jf-n[m])/times)*(span2)+n[m];
                for (int l = 0;l<(times-1);l++)
                {
                    intery[l] = round(times*intery2[l]);
                }
                for (int l=0;l<(times-1);l++)
                {
                    q1x[l] = q1L[interx[l]];
                    q2y[l] = q2LL[intery[l]];
                }
                Energy[m] = S(i-1,n[m])+pow(q1[i-1]- sqrt((jf-n[m])/times)*q2L[n[m]],2)+pow(norm(q1x-sqrt((jf-n[m])/times)*q2y,2),2);
            }

            double min_val = Energy.min(index);
            int loc = n[index];
            S(i,j) = min_val;
            ID(i,j) = loc;
        }
    }

    uvec tmp(2);
    int i = rownum;
    int j = colnum;
    tmp << i-1 << j-cut;
    start = max(tmp);
    tmp << j-1 << cut*(i-1);
    end = min(tmp);
    uvec n = start+seq_len(end-start+1);
    k = n.n_elem;
    interx = times*(i-1)+span1;
    span2 = conv_to<vec>::from(span1);
    vec Energy(k);

    for(int m = 0; m < k ; m++)
    {
        float jf = j;
        intery2 = ((jf-n[m])/times)*(span2)+n[m];
        for (int l= 0;l<(times-1);l++)
        {
            intery[l] = round(times*intery2[l]);
        }
        for (int l=0;l<(times-1);l++)
        {
            q1x[l] = q1L[interx[l]];
            q2y[l] = q2LL[intery[l]];
        }
        Energy[m]=S(i-1,n[m])+pow(q1[i-1]-sqrt((jf-n[m])/times)*q2L[n[m]],2)+pow(norm(q1x-sqrt((jf-n[m])/times)*q2y,2),2);
    }

    double min_val = Energy.min(index);
    int loc = n[index];
    S(i,j) = min_val;
    ID(i,j) = loc;

    uvec path(rownum);
    int count = ID(i,j);
    int oldcount;
    path(i-1) = count;

    while (count>1){
        i--;
        oldcount = count;
        count = ID(i,oldcount);
        path(i-1) = count;
    }

    dp_result out;
    out.J = path;
    out.NDist = S(rownum,colnum)/colnum;
    out.q2LL = q2LL;

    return out;
}

