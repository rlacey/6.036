hold on;
scatter(1,1,500,'r','fill');
scatter(3,1,500,'r','fill');
scatter(2,3,500,'g','fill');
plot(0:4, [2 2 2 2 2], 'black', 'linewidth',3)
scatter(0.75,2.5,500,'+','black');
scatter(0.75,1.5,500,'x','black');
hold off;
xlim([0, 4]);
ylim([0, 4]);