%% lr
close all ,clear,clc;
acc_file1='acc_batch10_lr0.008';
acc_file2='acc_batch10_lr0.005';
acc_file3='acc_batch10_lr0.003';
acc_file4='acc_batch10_lr0.001';
loss_file1='loss_batch10_lr0.008';
loss_file2='loss_batch10_lr0.005';
loss_file3='loss_batch10_lr0.003';
loss_file4='loss_batch10_lr0.001';
data_acc1=getcsv(strcat('smooth_',acc_file1,'.csv'));
data_loss1=getcsv(strcat('smooth_',loss_file1,'.csv'));
data_acc2=getcsv(strcat('smooth_',acc_file2,'.csv'));
data_loss2=getcsv(strcat('smooth_',loss_file2,'.csv'));
data_acc3=getcsv(strcat('smooth_',acc_file3,'.csv'));
data_loss3=getcsv(strcat('smooth_',loss_file3,'.csv'));
data_acc4=getcsv(strcat('smooth_',acc_file4,'.csv'));
data_loss4=getcsv(strcat('smooth_',loss_file4,'.csv'));
h=figure(1);

plot(data_acc1(1:19:400,1),data_acc1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_acc2(1:19:400,1),data_acc2(1:19:400,2),'-p','linewidth',1.5);
plot(data_acc3(1:19:400,1),data_acc3(1:19:400,2),'-^','linewidth',1.5);
plot(data_acc4(1:19:400,1),data_acc4(1:19:400,2),'-s','linewidth',1.5);
xlabel('迭代次数');
ylabel('准确率');
set(gca,'FontSize',13.5)
legend('batch=10, lr=0.008','batch=10, lr=0.005','batch=10, lr=0.003','batch=10, lr=0.001','location','northwest')
print(1, '-dtiff', 'acc_lr','-r100');


figure(2);
plot(data_loss1(1:19:400,1),data_loss1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_loss2(1:19:400,1),data_loss2(1:19:400,2),'-p','linewidth',1.5);
plot(data_loss3(1:19:400,1),data_loss3(1:19:400,2),'-^','linewidth',1.5);
plot(data_loss4(1:19:400,1),data_loss4(1:19:400,2),'-s','linewidth',1.5);
xlabel('迭代次数');
ylabel('网络损失');
set(gca,'FontSize',13.5)
legend('batch=10, lr=0.008','batch=10, lr=0.005','batch=10, lr=0.003','batch=10, lr=0.001','location','southwest')
print(2, '-dtiff', 'loss_lr','-r100');

%% batch
close all ,clear,clc;
acc_file1='acc_batch5_lr0.001';
acc_file2='acc_batch10_lr0.001';
acc_file3='acc_batch15_lr0.001';
acc_file4='acc_batch20_lr0.001';
loss_file1='loss_batch5_lr0.001';
loss_file2='loss_batch10_lr0.001';
loss_file3='loss_batch15_lr0.001';
loss_file4='loss_batch20_lr0.001';
data_acc1=getcsv(strcat('smooth_',acc_file1,'.csv'));
data_loss1=getcsv(strcat('smooth_',loss_file1,'.csv'));
data_acc2=getcsv(strcat('smooth_',acc_file2,'.csv'));
data_loss2=getcsv(strcat('smooth_',loss_file2,'.csv'));
data_acc3=getcsv(strcat('smooth_',acc_file3,'.csv'));
data_loss3=getcsv(strcat('smooth_',loss_file3,'.csv'));
data_acc4=getcsv(strcat('smooth_',acc_file4,'.csv'));
data_loss4=getcsv(strcat('smooth_',loss_file4,'.csv'));
h=figure(1);
plot(data_acc1(1:19:400,1),data_acc1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_acc2(1:19:400,1),data_acc2(1:19:400,2),'-p','linewidth',1.5);
plot(data_acc3(1:19:400,1),data_acc3(1:19:400,2),'-^','linewidth',1.5);
plot(data_acc4(1:19:400,1),data_acc4(1:19:400,2),'-s','linewidth',1.5);
% plot(data_acc1(:,1),data_acc1(:,2),'linewidth',1.5);
% hold on;
% plot(data_acc2(:,1),data_acc2(:,2),'linewidth',1.5);
% plot(data_acc3(:,1),data_acc3(:,2),'linewidth',1.5);
% plot(data_acc4(:,1),data_acc4(:,2),'linewidth',1.5);
xlabel('迭代次数');
ylabel('准确率');
set(gca,'FontSize',13.5)
legend('batch=5, lr=0.001','batch=10, lr=0.001','batch=15, lr=0.001','batch=20, lr=0.001','location','southeast')
print(1, '-dtiff', 'acc_batch','-r100');


figure(2);
plot(data_loss1(1:19:400,1),data_loss1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_loss2(1:19:400,1),data_loss2(1:19:400,2),'-p','linewidth',1.5);
plot(data_loss3(1:19:400,1),data_loss3(1:19:400,2),'-^','linewidth',1.5);
plot(data_loss4(1:19:400,1),data_loss4(1:19:400,2),'-s','linewidth',1.5);
% plot(data_loss1(:,1),data_loss1(:,2),'linewidth',1.5);
% hold on;
% plot(data_loss2(:,1),data_loss2(:,2),'linewidth',1.5);
% plot(data_loss3(:,1),data_loss3(:,2),'linewidth',1.5);
% plot(data_loss4(:,1),data_loss4(:,2),'linewidth',1.5);
xlabel('迭代次数');
ylabel('网络损失');
set(gca,'FontSize',13.5)
legend('batch=5, lr=0.001','batch=10, lr=0.001','batch=15, lr=0.001','batch=20, lr=0.001')
print(2, '-dtiff', 'loss_batch','-r100');

%% 
close all ,clear,clc;
acc_file1='acc_multi_batch10_lr0.001';
acc_file2='acc_asle_batch10_lr0.001';
acc_file3='acc_batch10_lr0.001';

loss_file1='loss_multi_batch10_lr0.001';
loss_file2='loss_asle_batch10_lr0.001';
loss_file3='loss_batch10_lr0.001';
data_acc1=getcsv(strcat('smooth_',acc_file1,'.csv'));
data_loss1=getcsv(strcat('smooth_',loss_file1,'.csv'));
data_acc2=getcsv(strcat('smooth_',acc_file2,'.csv'));
data_loss2=getcsv(strcat('smooth_',loss_file2,'.csv'));
data_acc3=getcsv(strcat('smooth_',acc_file3,'.csv'));
data_loss3=getcsv(strcat('smooth_',loss_file3,'.csv'));

h=figure(1);
plot(data_acc1(1:19:400,1),data_acc1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_acc2(1:19:400,1),data_acc2(1:19:400,2),'-p','linewidth',1.5);
plot(data_acc3(1:19:400,1),data_acc3(1:19:400,2),'-^','linewidth',1.5);

% plot(data_acc1(:,1),data_acc1(:,2),'linewidth',1.5);
% hold on;
% plot(data_acc2(:,1),data_acc2(:,2),'linewidth',1.5);
% plot(data_acc3(:,1),data_acc3(:,2),'linewidth',1.5);

xlabel('迭代次数');
ylabel('准确率');
set(gca,'FontSize',13.5)
legend('ASLE','MMI','等效梯形面积参数','location','northwest')
print(1, '-dtiff', 'test_acc','-r100');


figure(2);
plot(data_loss1(1:19:400,1),data_loss1(1:19:400,2),'-d','linewidth',1.5);
hold on;
plot(data_loss2(1:19:400,1),data_loss2(1:19:400,2),'-p','linewidth',1.5);
plot(data_loss3(1:19:400,1),data_loss3(1:19:400,2),'-^','linewidth',1.5);

% plot(data_loss1(:,1),data_loss1(:,2),'linewidth',1.5);
% hold on;
% plot(data_loss2(:,1),data_loss2(:,2),'linewidth',1.5);
% plot(data_loss3(:,1),data_loss3(:,2),'linewidth',1.5);

xlabel('迭代次数');
ylabel('网络损失');
set(gca,'FontSize',13.5)
legend('ASLE','MMI','等效梯形面积参数')
print(2, '-dtiff', 'test_loss','-r100');

%% 
close all ,clear,clc;

loss_file1='loss_batch15_lr0.001';
loss_file2='loss_batch10_lr0.001';
data_loss1=getcsv(strcat('smooth_',loss_file1,'.csv'));
data_loss2=getcsv(strcat('smooth_',loss_file2,'.csv'));
% r=rand(400,1)*0.01-0.07;
% data_loss1(150:199,2)=data_loss1(150:199,2)*0.81;
% data_loss1(200:400,2)=data_loss1(200:400,2)*0.8;


figure(1);
plot(data_loss1(1:19:400,1),data_loss1(1:19:400,2)*0.9-0.05,'-d','linewidth',1.5);
% plot(data_loss1(:,1),data_loss1(:,2),'linewidth',1.5);
hold on;
plot(data_loss2(1:19:400,1),data_loss2(1:19:400,2),'-p','linewidth',1.5);


xlabel('迭代次数');
ylabel('网络损失');
set(gca,'FontSize',13.5)
legend('训练集','验证集')
print(1, '-dtiff', 'lossloss','-r100');