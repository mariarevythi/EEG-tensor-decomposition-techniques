load("EEGData.mat")
%% temporal components
EEGt=permute(EEGData, [2 1 3]);
EEGt=reshape(EEGt, 700, 60*477);
size(EEGt)
%% PCA in time
[coefft,scoret,latentt,tsquaredt,explainedt,mut] = pca(EEGt');
figure;plot(coefft(:,1:2))

%% NMF in time
[Wt,Ht,Dt]=nnmf(EEGt,2);
figure;plot(Wt)

%% ICA in time
[Zt,Wicat,Tt]=fastICA(EEGt,2);
figure;plot(Wicat')

%% spatial components
EEGs=reshape(EEGData, 60, 700*477);
size(EEGs)
%% PCA  space
[coeff,score,latent,tsquared,explained,mu] = pca(EEGs');
figure;bar(coeff(:,1:2)')
%% NMF in space
[Wnmf,H,D]=nnmf(EEGs,2);
figure;bar(Wnmf')
%% ICA in space
[Z,Wica,T]=fastICA(EEGs,2);
figure;bar(Wica)
%% tensor decomp
EEG=tensor(EEGData);

%% Apply PARAFAC
[P, U0] = cp_als(EEG,2);
figure;subplot(3,1,1)
bar(P.U{1}')
subplot(3,1,2)
plot(P.U{2})
subplot(3,1,3)
plot(P.U{3})

%% Apply Non-negative PARAFAC
[Pn, U0n] = cp_nmu(EEG,2);
figure;subplot(3,1,1)
bar(Pn.U{1}')
subplot(3,1,2)
plot(Pn.U{2})
subplot(3,1,3)
plot(Pn.U{3})

%% Apply Tucker
[Pt, U0t] = tucker_als(EEG,[2 2 2]);
figure;subplot(3,1,1)
bar(Pt.U{1}')
subplot(3,1,2)
plot(Pt.U{2})
subplot(3,1,3)
plot(Pt.U{3})

%% Apply Non-negative Tucker
opts=[];
[A,C,Out] = ntd(EEG,[2 2 2],opts)
figure;subplot(3,1,1)
bar(A{1,1}')
subplot(3,1,2)
plot(A{1,2})
subplot(3,1,3)
plot(A{1,3})

 %%   Non-negative Tucker parameters
my_par1=A{1,3}(:,1);   %1= my first compoment
my_par2=A{1,3}(:,2);

my_par=[my_par1 my_par2];
% classify stim
[auc,dec,conf]=decoder(my_par,stim');
figure;imagesc(conf);colorbar
%% scatterplots
 %figure;gscatter(my_par1,my_par1,stim)
 figure;gscatter(my_par1,my_par2,stim)
 figure;gscatter (A{1,3}(:,1),A{1,3}(:,2),stim)
