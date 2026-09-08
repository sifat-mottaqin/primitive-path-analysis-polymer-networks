function [LN, rCenter] = calcGaussianLNandCoE(chain1, chain2)
% Computes Gaussian linking number between two open chains
% using the solid-angle summation
% Input: N1*3 and N2*3 
% Ouput: LN and [x y z] of CoE
N1 = size(chain1,1) - 1; % number of segments
N2 = size(chain2,1) - 1;
LN = 0;
for I = 1:N1
    rI   = chain1(I,:);
    rIp1 = chain1(I+1,:);
    for J = 1:N2
        rJ   = chain2(J,:);
        rJp1 = chain2(J+1,:);
        % Define vectors
        k = rI   - rJ;
        l = rIp1 - rJ;
        m = rIp1 - rJp1;
        n = rI   - rJp1;
        % two tetrahedra
        Omega1 = solid_angle(k, l, m);
        Omega2 = solid_angle(m, n, k);
        OmegaIJ = 2 * (Omega1 + Omega2);
        LN = LN + OmegaIJ;
    end
end
% Prefactor 1/(4π)
LN = LN / (4*pi);
rCenter = computeWritheCenter(chain1, chain2);
end

function Omega = solid_angle(a, b, c)
% Solid angle of triangle defined by a,b,c using
% Van Oosterom & Strackee formula
triple_prod = dot(a, cross(b, c));
denom = norm(a)*norm(b)*norm(c) + ...
    dot(a,b)*norm(c) + ...
    dot(c,a)*norm(b) + ...
    dot(b,c)*norm(a);
Omega = atan2(triple_prod, denom);
end

function rCenter = computeWritheCenter(chain1, chain2)
% Computes center of entanglement
% return [x y z]
    n1 = size(chain1,1)-1; n2 = size(chain2,1)-1;
    num = [0 0 0]; den = 0;
    for i = 1:n1
        a1 = chain1(i,:); 
        b1 = chain1(i+1,:); 
        d1 = b1-a1; 
        m1 = 0.5*(a1+b1);
        for j = 1:n2
            a2 = chain2(j,:); 
            b2 = chain2(j+1,:); 
            d2 = b2-a2; 
            m2 = 0.5*(a2+b2);
            R  = m1 - m2;   
            d  = norm(R);
            if d < 1e-6 
                continue; 
            end
            w  = dot(R, cross(d1,d2)) / d^3;
            num = num + w*0.5*(m1+m2);  
            den = den + w;
        end
    end
    if abs(den) < 1e-12
        rCenter = [NaN NaN NaN];
    else
        rCenter = num / den;
    end
end