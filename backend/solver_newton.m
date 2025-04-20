function [raiz, iteraciones] = solver_newton(f, df, x0, tol)
    iteraciones = 0;
    error = inf;
    while error > tol && iteraciones < 100
        fx = feval(f, x0);
        dfx = feval(df, x0);
        if dfx == 0
            error('Derivada cero');
        end
        x1 = x0 - fx / dfx;
        error = abs((x1 - x0) / x1);
        x0 = x1;
        iteraciones = iteraciones + 1;
    end
    raiz = x0;
end