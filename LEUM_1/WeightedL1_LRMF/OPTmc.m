% The cyclic weighted median algorithm for L1-norm matrix factorization %%%%%
% Solve
% min_{v}  ||W.*(Matrix-uv^T)||_1 .
% Please cite: "Deyu Meng, Zongben Xu, Lei Zhang, Ji Zhao. A cyclic
% weighted median method for L1 low-rank matrix factorization with missing entries. AAAI 2013."

% Written by: Deyu Meng, Xi'an Jiaotong University
% Email: dymeng@mail.xjtu.edu.cn
% Homepage: http://dymeng.gr.xjtu.edu.cn

function v = OPTmc(Matrix,W,u)
[d n] = size(Matrix);
Matrix = Matrix.*(sign(u)*ones(1,n));
u = abs(u);
M1 = Matrix.*(u.^(-1)*ones(1,n));
[A IND] = sort(W.*(M1-min(min(M1))+1));
[TIND] = find(A == 0);
Tu = u;
TI = Tu(IND);
TI(TIND) = 0;
SI = cumsum(TI);
TSUM = sum(TI)/2;
TM = sign(SI-ones(d,1)*TSUM);
TMM(1,:) = TM(1,:);
TMM(2:d,:) = TM(2:d,:)-TM(1:d-1,:);
TMM(2:d,:) = TMM(2:d,:).*TM(2:d,:);
IN = find(TMM>0);
v = A(IN)+min(min(M1))-1;