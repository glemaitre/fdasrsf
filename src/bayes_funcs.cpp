#include "armadillo"

arma::vec approx(int nd, arma::vec xd, arma::vec yd,int ni, arma::vec xi)
{
  int i,k;
  double t;
  arma::vec yi(ni);
  for ( i = 0; i < ni; i++ )
  {
    if ( xi(i) <= xd(0) )
    {
      t = ( xi(i) - xd(0) ) / ( xd(1) - xd(0) );
      yi(i) = ( 1.0 - t ) * yd(0) + t * yd(1);
    }
    else if ( xd(nd-1) <= xi(i) )
    {
      t = ( xi(i) - xd(nd-2) ) / ( xd(nd-1) - xd(nd-2) );
      yi(i) = ( 1.0 - t ) * yd(nd-2) + t * yd(nd-1);
    }
    else
    {
      for ( k = 1; k < nd; k++ )
      {
        if ( xd(k-1) <= xi(i) && xi(i) <= xd(k) )
        {
          t = ( xi(i) - xd(k-1) ) / ( xd(k) - xd(k-1) );
          yi(i) = ( 1.0 - t ) * yd(k-1) + t * yd(k);
          break;
        }
      }
    }
  }
  
  return yi;

}

arma::uvec seq_len(int n){
    arma::uvec out(n);
    out.fill(0);
    int k;

    for (k=0; k<n; k++){
        out[k] = k;
    }
    return out;
}
