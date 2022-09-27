function [X, cx] = ALMSDP(At, b, c, mb, p, tao, y, Y)
C = reshape(c, mb, mb);
sigma = 1e-4;
gama = 2;
MaxIter = 30;
tolgrad = 1e-8;
for iter = 1:MaxIter
    [Y, ~, ~] = SDP_ALM_subprog(At, b, c, C, mb, p, sigma, y, Y, tolgrad);
    X = Y*Y';
    x = X(:);
    cx = x'*c;
    Axb = At'*x - b;
    y = y - sigma*Axb;
    yA = reshape(y'*At', mb, mb);
    DfX = C - yA;
    lamda = diag(DfX*X);
    S = DfX - diag(lamda);
    [v, mineigS] = eigs(S, 1, 'smallestreal');
%     by = b'*y + sum(lamda);
%     gap = abs(cx-by)/abs(cx+by);
%     e = sort(diag(qr(Y)),'descend');
    r = 1;
%     while r < p && e(r+1) > 1e-5*e(1) && e(r+1)/e(r) < 10
%         r = r + 1;
%     end
%     if r < p
        [q, ~] = eigs(Y'*Y, 1, 'smallestreal');
%     end
%     if r == p - 1
        U = v*q(:,1)';
%     elseif r < p - 1
%         p = r + 1;
%         [V,D,~] = svd(Y);
%         Y = V(:,1:p)*D(1:p,1:p);
%         U = v*q(1:p,1)';
%     else
%         U = [zeros(mb,p) v];
%         Y = [Y zeros(mb,1)];
%         p = p + 1;
%     end
    Y = Y + 0.1*U;
    for i = 1:mb
        Y(i,:) = Y(i,:)/norm(Y(i,:));
    end
   disp(['ALM iteration ' num2str(iter) ': fval = ' num2str(cx,10) ', rank X = ' num2str(r) ', mineigS = ' num2str(mineigS) ', p = ' num2str(p)]);
   if max([norm(Axb),abs(mineigS)]) < tao
       break;
   else
        sigma = min(sigma*gama, 1e4);
   end
end
end