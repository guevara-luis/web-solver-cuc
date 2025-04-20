function raiz = solver_bisection(f, a, b, tol)
    while (b - a)/2 > tol
        c = (a + b)/2;
        if feval(f, c) == 0
            break;
        elseif sign(feval(f, c)) == sign(feval(f, a))
            a = c;
        else
            b = c;
        end
    end
    raiz = (a + b)/2;
end