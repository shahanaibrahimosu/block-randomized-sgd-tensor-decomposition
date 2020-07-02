function H = proxr( Hb, ops, d)
    switch ops.constraint{d}
        case 'nonnegative'
            H =  max(0 , Hb) ;
        case 'simplex_col'
            H = ProjectOntoSimplex(Hb, ops.rho);
        case 'simplex_row'
            H = ProjectOntoSimplex(Hb', 1);
            H = H';
        case 'l1'
            H = sign( Hb ) .* max( 0, abs(Hb) - (ops.l1{d}/rho) );
        case 'l1n'
            H = max( 0, Hb - ops.l1{d}/rho );
        case 'l2'
            H = ( rho/(ops.l2{d}+rho) ) * Hb;
        case 'l2n'
            H = ( rho/(ops.l2{d}+rho) ) * max(0,Hb);
        case 'l2-bound'
           nn = sqrt( sum( Hb.^2 ) );
            H = Hb * diag( 1./ max(1,nn) );
        case 'l2-boundn'
            H = max( 0, Hb );
           nn = sqrt( sum( H.^2 ) );
            H = H * diag( 1./ max(1,nn) );
        case 'l0'
            T = sort(Hb,2,'descend');
            t = T(:,4); T = repmat(t,1,size(T,2));
            H = Hb .* ( Hb >= T );
         case 'noconstraint'
            H = Hb;
    end
end