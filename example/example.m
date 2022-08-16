clc; 
clear; 
close all; 
pgdpath   = '../../STRIDE';
addpath(genpath(pgdpath));
%% Generate random binary quadratic program
d       = 20; % BQP with d variables
x       = msspoly('x',d); % symbolic decision variables using SPOTLESS
Q       = rand(d); Q = (Q + Q')/2; % a random symmetric matrix
% e       = rand(d,1);
f       = x'*Q*x; % objective function of the BQP
h       = x.^2 - 1; % equality constraints of the BQP (binary variables)
g       = []; % ask the first variable to be positive

%% Relax BQP into an SDP
problem.vars            = x;
problem.objective       = f;
problem.equality        = h; 
problem.inequality      = g;
kappa                   = 2; % relaxation order
% basis = [1;x;monomials(x,2:kappa)];
[SDP,info]              = dense_sdp_relax(problem,kappa);
SDP.M       = length(info.v); % upper bound on the trace of the moment matrix
% need the following for fast computation in the local search method
info.v      = msspoly2degcoeff(info.v);
info.f      = msspoly2degcoeff(info.f);
info.J      = msspoly2degcoeff(info.J);
At = SDP.sedumi.At;
b = SDP.sedumi.b;
c = SDP.sedumi.c;
K = SDP.sedumi.K;
Nx = K.s;

%% Generate SOS data
sx = sym('x', [1 d]);
sf       = sx*Q*sx.';
sh = sx.^2 - 1;
[sA, sB, sb] = SOStoSDP(sf, sh, sx, kappa);
mb = size(sA{1},1);
vmb = mb*(mb+1)/2;
ssA = sparse(length(sA),vmb);
for i = 1:length(sA)
    ssA(i, :) = Mat2Vec(sA{i})';
end
sB1 = [ssA sB];
dA = zeros(length(sA),1);
for i = 1:length(sA)
    dA(i) = 1/sum(ssA(i,:).^2);
end
iD = sparse(1:length(sA),1:length(sA),dA);
iA = iD - iD*sB*(sparse(1:size(sB,2),1:size(sB,2),ones(size(sB,2),1))+sB'*iD*sB)^(-1)*sB'*iD;
M1 = sparse(1:size(sB1,2),1:size(sB1,2),ones(size(sB1,2),1)) - sB1'*iA*sB1;
M2 = sB1'*iA;

%% Solve using STRIDE
sdpnalpath  = '../../SDPNAL+v1.0';
pgdopts.pgdStepSize     = 10;
pgdopts.SDPNALpath      = sdpnalpath;
pgdopts.tolADMM         = 10e-5;
pgdopts.phase1          = 1;
pgdopts.rrOpt           = 1:3;
pgdopts.rrFunName       = 'local_search_bqp'; % see solvers/local_search_bqp.m for implementation of local search
pgdopts.rrPar           = info; % need the original POP formulation for local search
pgdopts.maxiterLBFGS    = 1000;
pgdopts.maxiterSGS      = 300;
pgdopts.tolLBFGS        = 1e-12;
pgdopts.tolPGD          = 1e-8;

[outPGD,sXopt,syopt,sSopt]     = PGDSDP(SDP.blk, SDP.At, SDP.b, SDP.C, [], pgdopts);
time_pgd                    = outPGD.totaltime;
% round solutions and check optimality certificate
% res = get_performance_bqp(Xopt,yopt,Sopt,SDP,info,pgdpath);

%% Solve using MOSEK
% tic
% prob       = convert_sedumi2mosek(At, b,c,K);
% [~,res]    = mosekopt('minimize echo(0)',prob);
% [Xopt,yopt,Sopt,obj] = recover_mosek_sol_blk(res,SDP.blk);
% tmosek = toc;
% figure; bar(eig(Xopt{1}));

%% Solve using Manopt
m = length(b);
p = 2;
A = At';
C = reshape(c, Nx, Nx);
options.maxtime = inf;
flag = 0;

while flag == 0
tic
[Y, fval, info] = SDP_AdptvALM_subprog(A, At, b, C, c, Nx, m, p, options);
% X = Y'*Y;
tmanipop = toc;

%% Solve using fmincon
% fobj = @(y) y'*Q*y + e'*y;
% [px,cv] = fmincon(fobj,zeros(d,1),[],[],[],[],[],[],@binary)

%% ALM方法参数设置
% sigma = 1e-3;
% gama = 13;
% % Y = full(msubs(basis, x, px));
% Y = [];
% % Y = [Y zeros(size(Y,1),1)];
% yk = zeros(m,1);

%% 迭代循环
% tic
% MaxIter = 20;
% for iter = 1:MaxIter
%     [Y, fval, info] = SDP_ALM_subprog(A, At, b, C, c, Nx, p, sigma, yk, Y);
%     X = Y*Y';
%     z = X(:);
%     cx = z'*c;
%     Axb = (z'*At)' - b;
%     if norm(Axb) < 1e-4
%         break;
%     else
%         disp(['Iter ' num2str(iter) ': fval = ' num2str(cx,10)]);
%         yk = yk + 2*Axb*sigma;
%         sigma = min(sigma*gama, 1e4);
%     end
% end
% fval = cx;
% tmanipop = toc;

% disp(['Mosek: ' num2str(tmosek) 's'])
disp(['ManiPOP: ' num2str(tmanipop) 's'])
disp(['Stride: ' num2str(time_pgd) 's'])
% disp(['Mosek: ' num2str(obj(1))])
disp(['ManiPOP: ' num2str(fval)])
disp(['Stride: ' num2str(outPGD.pobj)])

% [V,D] = eig(X);
% temp = zeros(length(sA), mb-1);
% for i = 1:length(sA)
%     for j = 1:mb-1
%         temp(i, j) = trace(sA{i}*V(:,j)*V(j,:));
%     end
% end
% sB0 = [temp sB];
% sol = sB0\sb;

% for i = 1:100
%     psol = sol;
%     psol(1:mb-1) = max(0,sol(1:mb-1));
%     lsol = 2*psol - sol;
%     lsol = lsol - sB'*(sB*sB')^(-1)*(sB*lsol-sb);
%     sol = sol + 1*(lsol - psol);
%     minEig = min(lsol(1:mb-1));
%     error = norm(psol - lsol);
%     disp(['step ' num2str(i) '  error:' num2str(error) ', minEig:' num2str(minEig)]);
% end

tic
% psd = V*diag([sol(1:mb-1);0])*V';
% sol = [Mat2Vec(psd); sol(mb:end)];
psd = zeros(mb, mb);
sol = zeros(vmb+size(sB,2), 1);
% psd = rand(mb); 
% psd = (psd + psd')/2;
% sol = [Mat2Vec(psd); rand(size(sB,2),1)];
ssb = sb;
ssb(1) = ssb(1) - fval;
ssb = M2*ssb;
gap = 1;
i = 1;
while gap > 1e-2 && i <= 200
    [V,D] = eig(psd);
    psd = V*diag(max(0,diag(D)))*V';
    psol = sol;
    psol(1:vmb) = Mat2Vec(psd);
    lsol = 2*psol - sol;
    lsol = M1*lsol + ssb;
    sol = sol + 1*(lsol - psol);
    psd = Vec2Mat(sol(1:vmb), mb);
    minEig = min(eig(Vec2Mat(lsol(1:vmb), mb)));
    gap = - minEig*mb/abs(fval);
    % error = norm(psol - lsol);
    % disp(['Step ' num2str(i) ': error = ' num2str(error) ', gap <= ' num2str(gap)]);
    % disp(['Step ' num2str(i) ': gap <= ' num2str(gap)]);
    i = i + 1;
end
tcert = toc;
disp(['Certify global optimality: ' num2str(tcert) 's']);
if gap <= 1e-2
    flag = 1;
    disp(['Global optimality certified!']);
else
    disp(['Global optimality not certified, use another initial point.']);
%     p = p + 1;
end
end

%% Yalmip
% yx = sdpvar(d,1);
% sdpvar lower;
% p = yx'*Q*yx;
% yh = yx.^2 - 1;
% [s1,c1] = polynomial(yx,2);
% [s2,c2] = polynomial(yx,2);
% F = [sos(p-lower-[s1 s2]*yh)];
% [sol,v,sQ] = solvesos(F,-lower,[],[c1;c2;lower]);
% sQ{1} = sQ{1}([1;3;2;6;5;4],[1;3;2;6;5;4]);
% tt = zeros(15,1);
% for i = 1:15
%     tt(i) = trace(sQ{1}*sA{i});
% end
% norm(tt + sB*[value(c1);value(c2)] - sb)
% vx = monolist(yx, 2);
% clean(v{1}'*sQ{1}*v{1} + vx'*value(c1)*yh(1) + vx'*value(c2)*yh(2) + value(lower) - p,1e-6)

%% helper functions
function s = msspoly2degcoeff(f)
[~,degmat,coeff,~] = decomp(f);
s.degmat = degmat';
s.coefficient = coeff;
end
