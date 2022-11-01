function  [Y, fval, info] = SDP_ALM_subprog0(At, b, c, C, n, p, sigma, y, Y0, U, tolgrad)
    % Pick the manifold of n-by-p matrices with unit norm rows.
    % manifold = obliquefactory(p, n, true);
    manifold = euclideanfactory(n, p);
    % manifold = elliptopefactory(n, p);
    % manifold = symfixedrankYYfactory(n, p);
    % manifold = spectrahedronfactory(n, p);
    problem.M = manifold;
    
    % Define the cost function to be /minimized/.
    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        X = Y*Y';
        x = X(:);
        cx = x'*c;
        Axb = At'*x - b - y/sigma;
        f = cx + sigma/2*(Axb'*Axb);
        AxbA = Axb'*At';
        yA = reshape(AxbA, n, n);
        S = C + sigma*yA;
        store.G = 2*S*Y;
        store.S = S;
    end
    
    % Define the Riemannian gradient.
    problem.grad = @grad;
    function [G, store] = grad(Y, store)
%         X = Y*Y';
%         x = X(:);
%         Axb = At'*x - b - y/sigma;
%         AxbA = Axb'*At';
%         yA = reshape(AxbA, n, n);
%         S = C + sigma*yA;
%         G = 2*S*Y;
        G = store.G;
    end

    % If you want to, you can specify the Riemannian Hessian as well.
    problem.hess = @hess;
    function [H, store] = hess(Y, Ydot, store)
%         X = Y*Y';
%         x = X(:);
%         Axb = At'*x - b - y/sigma;
%         AxbA = Axb'*At';
%         yA = reshape(AxbA, n, n);
%         S = C + sigma*yA;
        S = store.S;
        H = 2*S*Ydot;
        Xdot = Ydot*Y';
        xdot = Xdot(:);
        AxbdotA = xdot'*At*At';
        yAdot = reshape(AxbdotA, n, n);
        H = H + 4*sigma*(yAdot*Y);
    end

    % Call your favorite solver.
    opts = struct();
    opts.verbosity = 0;      % Set to 0 for no output, 2 for normal output
    opts.maxinner = 20;     % maximum Hessian calls per iteration
    opts.mininner = 5;
    opts.tolgradnorm = tolgrad; % tolerance on gradient norm
    opts.maxiter = 4;
%     if ~isempty(U)
%          g = getGradient(problem, Y0);
%          h = getHessian(problem, Y0, U);
%          store = [];
%          [g, store] = egrad(Y0, store);
%          [h, store] = ehess(Y0, U, store);
%         disp(['grad: ' num2str(trace(g'*U)) ', hess: ' num2str(trace(U'*h))]);
%     end
    [Y, fval, info] = trustregions(problem, Y0, opts);
%     figure;
%     semilogy([info.iter], [info.gradnorm], '.-');
%     xlabel('Iteration number');
%     ylabel('Norm of the gradient of f');
end
