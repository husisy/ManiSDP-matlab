% spotpath   = '../../../Programs/spotless';
% addpath(genpath(spotpath));
% pgdpath   = '../../STRIDE';
% addpath(genpath(pgdpath));
% sdpnalpath  = '../../SDPNAL+v1.0';

%% Generate random quartic program
rng(1);
n = 18;
I{1} = [1;2;3;4;5;6;7;8;9;10];
I{2} = [9;10;11;12;13;14;15;16;17;18];
t = length(I);
sp = [];
for i = 1:t
    sp = [sp get_basis(n, 4, I{i})];
end
sp = unique(sp', 'rows');
coe = randn(size(sp, 1), 1);
[At, b, c, K] = qsmom_sparse(n, I, coe);

%% Solve using ManiSDP
rng(0);
clear options;
options.tol = 1e-8;
tic
[~, fval, data] = ManiSDP_multiblock(At, b, c, K, options);
emani = max([data.gap, data.pinf, data.dinf]);
tmani = toc;

% fz = [[1:length(data.fac_size)]' data.fac_size];
% residue = [[1:length(data.seta)]' log10(data.seta)];
% writematrix(fz, 'd://works/mypaper/manisdp/qs_fz_60.txt','Delimiter',' ');
% writematrix(residue, 'd://works/mypaper/manisdp/qs_residue_60.txt','Delimiter',' ');

%% Relax QP into an SDP
% problem.vars            = x;
% problem.objective       = f;
% problem.equality        = h; 
% kappa                   = 2; % relaxation order
% [SDP,info]              = dense_sdp_relax(problem,kappa);
% At = SDP.sedumi.At;
% b = SDP.sedumi.b;
% c = SDP.sedumi.c;
% K = SDP.sedumi.K;
% mb = K.s;
% C = full(reshape(c, mb, mb));

%% Solve using STRIDE
% SDP.M       = 3; % upper bound on the trace of the moment matrix
% info.v      = msspoly2degcoeff(info.v);
% info.f      = msspoly2degcoeff(info.f);
% info.J      = msspoly2degcoeff(info.J);
% pgdopts.pgdStepSize     = 10;
% pgdopts.SDPNALpath      = sdpnalpath;
% pgdopts.tolADMM         = 1e-4;
% pgdopts.phase1          = 1;
% pgdopts.rrOpt           = 1:3;
% pgdopts.rrFunName       = 'local_search_q4s'; % see solvers/local_search_bqp.m for implementation of local search
% pgdopts.rrPar           = info; % need the original POP formulation for local search
% pgdopts.maxiterLBFGS    = 1000;
% pgdopts.maxiterSGS      = 300;
% pgdopts.tolLBFGS        = 1e-12;
% pgdopts.tolPGD          = 1e-8;
% 
% rng(0);
% tic
% [outPGD,X,y,S]     = PGDSDP(SDP.blk, SDP.At, SDP.b, SDP.C, [], pgdopts);
% by = b'*y;
% gap = abs(outPGD.pobj-by)/(abs(by)+abs(outPGD.pobj)+1);
% x = X{1}(:);
% eta = norm(At'*x - b)/(1+norm(b));
% [~, dS] = eig(S{1}, 'vector');
% mS = abs(min(dS))/(1+dS(end));
% epgd = max([gap, eta, mS]);
% time_pgd = toc;

%% Solve using MOSEK
% prob       = convert_sedumi2mosek(At, b, c, K);
% tic
% [~,res]    = mosekopt('minimize echo(3)',prob);
% [X,y,S,mobj] = recover_mosek_sol_blk(res, SDP.blk);
% by = b'*y;
% gap = abs(mobj(1)-by)/(abs(by)+abs(mobj(1))+1);
% x = X{1}(:);
% eta = norm(At'*x - b)/(1+norm(b));
% [~, dS] = eig(S{1}, 'vector');
% mS = abs(min(dS))/(1+dS(end));
% emosek = max([eta, gap, mS]);
% tmosek = toc;

%% Solve using COPT
% warning('off');
% tic
% X0 = sdpvar(mb, mb, 'hermitian', 'real');
% F = [X0 >= 0, At'*X0(:) == b];
% obj = c'*X0(:);
% opts = sdpsettings('verbose', 1, 'solver', 'copt');
% sol = optimize(F, obj, opts);
% vcopt = value(obj);
% X = value(X0);
% S = dual(F(1));
% y = dual(F(2));
% by = - b'*y;
% gap = abs(vcopt-by)/(abs(by)+abs(vcopt)+1);
% x = X(:);
% eta = norm(At'*x - b)/(1+norm(b));
% [~, dS] = eig(S, 'vector');
% mS = abs(min(dS))/(1+dS(end));
% ecopt = max([eta, gap, mS]);
% tcopt = toc;

%% Solve using SDPLR
% rng(0);
% pars.printlevel = 1;
% pars.feastol = 1e-8;
% tic
% [x,y] = sdplr(At', b, c, K, pars);
% vlr = c'*x;
% S = C - reshape(At*y, mb, mb);
% by = b'*y;
% gap = abs(vlr-by)/(abs(by)+abs(vlr)+1);
% eta = norm(At'*x - b)/(1+norm(b));
% [~, dS] = eig(S, 'vector');
% mS = abs(min(dS))/(1+dS(end));
% elr = max([eta, gap, mS]);
% tlr = toc;

%% Solve using SDPNAL+
% options.tol = 1e-8;
% addpath(genpath(sdpnalpath));
% rng(0);
% tic
% [objnal,X,~,y,S] = sdpnalplus(SDP.blk, SDP.At, SDP.C, SDP.b, [], [], [], [], [], options);
% by = b'*y;
% gap = abs(objnal(1)-by)/(abs(by)+abs(objnal(1))+1);
% x = X{1}(:);
% eta = norm(At'*x - b)/(1+norm(b));
% [~, dS] = eig(S{1}, 'vector');
% mS = abs(min(dS))/(1+dS(end));
% enal = max([eta, gap, mS]);
% tnal = toc;

% fprintf('Mosek: optimum = %0.8f, eta = %0.1e, time = %0.2fs\n', mobj(1), emosek, tmosek);
% fprintf('SDPLR: optimum = %0.8f, eta = %0.1e, time = %0.2fs\n', vlr, elr, tlr);
% fprintf('SDPNAL: optimum = %0.8f, eta = %0.1e, time = %0.2fs\n', objnal(1), enal, tnal);
% fprintf('Stride: optimum = %0.8f, eta = %0.1e, time = %0.2fs\n', outPGD.pobj, epgd, time_pgd);
fprintf('ManiSDP: optimum = %0.8f, eta = %0.1e, time = %0.2fs\n', fval, emani, tmani);
