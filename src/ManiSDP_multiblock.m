% This function solves general linear SDPs.
%  Min  <C, X>
%  s.t. A(X) = b,
%       X in S_+^{n_1×...×n_t}

function [X, obj, data] = ManiSDP_multiblock(At, b, c, K, options)

n = K.s;
nb = length(n);
if ~isfield(options,'p0'); options.p0 = ones(nb,1); end
if ~isfield(options,'AL_maxiter'); options.AL_maxiter = 100; end
if ~isfield(options,'gama'); options.gama = 2; end
if ~isfield(options,'sigma0'); options.sigma0 = 1e-2; end
if ~isfield(options,'sigma_min'); options.sigma_min = 1e-2; end
if ~isfield(options,'sigma_max'); options.sigma_max = 1e7; end
if ~isfield(options,'tol'); options.tol = 1e-8; end
if ~isfield(options,'theta'); options.theta = 1e-2; end
if ~isfield(options,'delta'); options.delta = 6; end
if ~isfield(options,'alpha'); options.alpha = 0.06; end
if ~isfield(options,'tolgradnorm'); options.tolgrad = 1e-8; end
if ~isfield(options,'TR_maxinner'); options.TR_maxinner = 40; end
if ~isfield(options,'TR_maxiter'); options.TR_maxiter = 4; end
if ~isfield(options,'tao'); options.tao = 0.01; end
if ~isfield(options,'line_search'); options.line_search = 0; end
if ~isfield(options,'solver'); options.solver = 0; end

fprintf('ManiSDP is starting...\n');
fprintf('SDP size: n = %i, m = %i\n', max(n), size(b,1));

A = At';
p = options.p0;
sigma = options.sigma0;
gama = options.gama;
y = zeros(length(b),1);
normb = 1 + norm(b);
x = zeros(sum(n.^2), 1);
YU = zeros(sum(n.^2), 1);
Y = [];
U = [];
% fac_size = [];
% seta = [];
problem.cost = @cost;
problem.grad = @grad;
problem.hess = @hess;
opts.verbosity = 0;     % Set to 0 for no output, 2 for normal output
opts.maxinner = options.TR_maxinner;     % maximum Hessian calls per iteration
opts.maxiter = options.TR_maxiter;
opts.tolgradnorm = options.tolgrad;

data.status = 0;
for i = 1:nb
    M.(['M' num2str(i)]) = euclideanfactory(n(i), p(i));
end
elems = fieldnames(M);
timespend = tic;
for iter = 1:options.AL_maxiter
%     fac_size = [fac_size; p];
    problem.M = productmanifold(M);
    if ~isempty(U)
        Y = line_search(Y, U);
    end
    if options.solver == 0
        [Y, ~, info] = trustregions(problem, Y, opts);
    elseif options.solver == 1
        [Y, ~, info] = arc(problem, Y, opts);
    elseif options.solver == 2
        [Y, ~, info] = steepestdescent(problem, Y, opts);
    elseif options.solver == 3
        [Y, ~, info] = conjugategradient(problem, Y, opts);
    elseif options.solver == 4
        [Y, ~, info] = barzilaiborwein(problem, Y, opts);
    elseif options.solver == 5
        [Y, ~, info] = rlbfgs(problem, Y, opts);
    else
        fprintf('Solver is not supported!\n');
        return;
    end
    gradnorm = info(end).gradnorm;
    ind = 1;
    for i = 1:nb
        X{i} = Y.(elems{i})*Y.(elems{i})';
        x(ind:ind+n(i)^2-1) = X{i}(:);
        ind = ind + n(i)^2;
    end
    obj = c'*x;
    Axb = A*x - b;
    pinf = norm(Axb)/normb;
    y = y - sigma*Axb;
    cy = c - At*y;
    dinfs = zeros(nb, 1);
    ind = 1;
    for i = 1:nb
        S{i} = reshape(cy(ind:ind+n(i)^2-1), n(i), n(i));
        ind = ind + n(i)^2;
        [vS{i}, dS{i}] = eig(S{i}, 'vector');
        dinfs(i) = max(0, -dS{i}(1))/(1+abs(dS{i}(end)));
    end
    dinf = max(dinfs);
    by = b'*y;
    gap = abs(obj-by)/(abs(by)+abs(obj)+1);
    fprintf('Iter %d, obj:%0.8f, gap:%0.1e, pinf:%0.1e, dinf:%0.1e, gradnorm:%0.1e, p_max:%d, sigma:%0.3f, time:%0.2fs\n', ...
             iter,    obj,       gap,       pinf,       dinf,   gradnorm,  max(p),  sigma,   toc(timespend));
    eta = max([pinf, gap, dinf]);
%     seta = [seta; eta];
    if eta < options.tol
        fprintf('Optimality is reached!\n');
        break;
    end
    if mod(iter, 10) == 0
        if iter > 20 && gap > gap0 && pinf > pinf0 && dinf > dinf0
            data.status = 2;
            fprintf('Slow progress!\n');
            break;
        else
            gap0 = gap;
            pinf0 = pinf;
            dinf0 = dinf;
        end
    end
    for i = 1:nb
        [V, D, ~] = svd(Y.(elems{i}));
        if size(D, 2) > 1
            e = diag(D);
        else
            e = D(1);
        end
        r = sum(e > options.theta*e(1)); 
        if r <= p(i) - 1         
             Y.(elems{i}) = V(:,1:r)*diag(e(1:r));
             p(i) = r;
        end
        nne = min(sum(dS{i} < 0), options.delta);
        if options.line_search == 1
            U.(elems{i}) = [zeros(n(i), p(i)) vS{i}(:,1:nne)];
        end
        p(i) = p(i) + nne;
        if options.line_search == 1
            Y.(elems{i}) = [Y.(elems{i}) zeros(n(i), nne)];
        else
            Y.(elems{i}) = [Y.(elems{i}) options.alpha*vS{i}(:,1:nne)];
        end
    end
    if pinf < options.tao*gradnorm
          sigma = max(sigma/gama, options.sigma_min);
    else
          sigma = min(sigma*gama, options.sigma_max);
    end
%    tolgrad = pinf;
end
data.S = S;
data.y = y;
data.gap = gap;
data.pinf = pinf;
data.dinf = dinf;
data.gradnorm = gradnorm;
data.time = toc(timespend);
% data.fac_size = fac_size;
% data.seta = seta;
if data.status == 0 && eta > options.tol
    data.status = 1;
    fprintf('Iteration maximum is reached!\n');
end

fprintf('ManiSDP: optimum = %0.8f, time = %0.2fs\n', obj, toc(timespend));

%         function Y = line_search(Y, U, t)
%             X = Y*Y';
%             D = U*U';
%             YU = Y*U' + U*Y';
%             q0 = A*X(:) - b;
%             q1 = A*YU(:);
%             q2 = A*D(:);
%             aa = sigma/2*norm(q2)^2;
%             bb = sigma*q1'*q2;
%             cc = c'*D(:) - (y - sigma*q0)'*q2 + sigma/2*norm(q1)^2;
%             dd = c'*YU(:) - (y - sigma*q0)'*q1;
%             alpha_min = 0.02;
%             alpha_max = 0.5;
%             sol = vpasolve(4*aa*t^3 + 3*bb*t^2 + 2*cc*t + dd == 0, t, [alpha_min alpha_max]);
%             alpha = [alpha_min;eval(sol);alpha_max];
%             [~,I] = min(aa*alpha.^4 + bb*alpha.^3 + cc*alpha.^2 + dd*alpha);
%             Y = Y + alpha(I)*U;
%         end

%         function Y = line_search(Y, U)
%              alpha = [0.02;0.04;0.06;0.08;0.1;0.2];
%              val = zeros(length(alpha),1);
%              for i = 1:length(alpha)
%                 val(i) = co(Y + alpha(i)*U);
%              end
%              [~,I] = min(val);
%              Y = Y + alpha(I)*U;
%         end

    function nY = line_search(Y, U)
         alpha = 0.2;
         cost0 = co(Y);
         i = 1;
         nY = Y + alpha*U;
         while i <= 15 && co(nY) - cost0 > -1e-3
              alpha = 0.8*alpha;
              nY = Y + alpha*U;
              i = i + 1;
         end
    end

   function val = co(Y)
        ind = 1;
        for i = 1:nb
            X{i} = Y.(elems{i})*Y.(elems{i})';
            x(ind:ind+n(i)^2-1) = X{i}(:);
            ind = ind + n(i)^2;
        end
        Axb = A*x - b - y/sigma;
        val = c'*x + sigma/2*(Axb'*Axb);
   end

   function [f, store] = cost(Y, store)
        ind = 1;
        for i = 1:nb
            X{i} = Y.(elems{i})*Y.(elems{i})';
            x(ind:ind+n(i)^2-1) = X{i}(:);
            ind = ind + n(i)^2;
        end
        Axb = A*x - b - y/sigma;
        f = c'*x + sigma/2*(Axb'*Axb);
    end
    
    function [G, store] = grad(Y, store)
        tt = c + sigma*At*Axb;
        ind = 1;
        for i = 1:nb
            S{i} = reshape(tt(ind:ind+n(i)^2-1), n(i), n(i));
            G.(elems{i}) = 2*S{i}*Y.(elems{i});
            ind = ind + n(i)^2;
        end
    end

    function [H, store] = hess(Y, U, store)
        ind = 1;
        for i = 1:nb
            T = U.(elems{i})*Y.(elems{i})';
            YU(ind:ind+n(i)^2-1) = T(:);
            ind = ind + n(i)^2;
            H.(elems{i}) = 2*S{i}*U.(elems{i});
        end
        AyU = YU'*At*A;
        ind = 1;
        for i = 1:nb
             H.(elems{i}) = H.(elems{i}) + 4*sigma*reshape(AyU(ind:ind+n(i)^2-1), n(i), n(i))*Y.(elems{i});
             ind = ind + n(i)^2;
        end
    end
end
