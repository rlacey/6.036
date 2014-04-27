iter = [1 2 3 4];
dataA = [-12349648.2363
-12073983.0009
-11975392.0625
-11932227.8478];

dataB = [0.71348296571
0.740204245427
0.752331663576
0.753987767415];

dataC = [-22278567.2639
-21614389.755 
-21319755.7617
-21198638.9397];

plot(iter, dataA, iter, dataB, iter, dataC);
xlabel('Iteration')
ylabel('Log-Likelihood')
legend('A', 'B', 'C')