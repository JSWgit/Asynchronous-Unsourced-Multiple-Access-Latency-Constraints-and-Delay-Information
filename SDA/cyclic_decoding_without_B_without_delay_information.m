clear
tic;
syms cc(w,m)
cc(w,m) = nchoosek(w,m);
% err=zeros(1,ts_max);
pdb=-25:1e-1:-15;
alp=0.2;
calp=1-alp;thr=100;
n=38400;
logm=128*log(2);
ka1=300:-20:20;
ka1=[ka1,1];
finalPewithB=zeros(length(ka1),length(pdb));
fianlPewithoutB=zeros(length(ka1),length(pdb));
counter=zeros(length(ka1),length(pdb));
target_pe_without_B=1e-30;
stepsize=1e-3;
tt=[0.255:1e-2:20];
ts=zeros(length(tt),1/stepsize+1);
for z=1:length(tt)
    t=tt(z);
    ts(z,1:length(0:stepsize:1/4/t))=0:stepsize:1/4/t;
end
%%
%modify: using err of the case without B to find the required power for without B
%cases, then find the corresponding error prob. of the case with B. should
%be able to reduce the simulation time.
%%
stpoint=length(pdb);
for kaind=1:length(ka1)

% kaind=length(ka1);
    ka=ka1(kaind)
    %%
    %%
    for pp=stpoint:-1:1
        p=10^(pdb(pp)/10);
        g2=thr*ones(1,ka);
        T_star2=zeros(1,ka);
        gb2=zeros(1,ka);
        for s=ka:-1:1
            g_bar=thr*ones(ka,ka);
            T_star=ones(ka,ka);gb=ones(1,ka-s);
            logthe=log(cc(exp(logm)-ka,s))+s*log(1+alp*n);%To investigate the effect from the unknown delay information and the noise amplifier

            C2=double(-logthe+n/2*log(1+2*s*p*tt))';
            bias22=-C2+alp*n*s*p*tt'.*(8*tt'.*ts+4*tt'-1)./(1+2*s*p*tt'.*(1+ts).*(1-4*tt'.*ts))+calp*n*s*p*tt'.*(4*tt'.*ts+2*tt'-1)./(1+2*s*p*tt'.*(1+ts).*(1-2*tt'.*ts));
            ind_possible=find(bias22(:,2)<=0);
            bias22(bias22(:,2)>0,:)=[];
            remove_term=sum(bias22>0,2)>=1;
            bias22=bias22(remove_term,:);
            ind_possible=ind_possible(remove_term);
            lengthofpossibleterms=length(bias22(:,1));
            clear ts_ind_temp2
            if lengthofpossibleterms==0
                g2(s)=100;
                continue
            end
            bias22(bias22<0)=inf;
            [~,ts_ind_temp2]=min(bias22,[],2);

            %%
            ts_temp=ts(1,ts_ind_temp2);
            g2_temp=-ts_temp.*C2(ind_possible)'+alp*n/2*log((1+2*s*p*tt(ind_possible))./(1+2*s*p*tt(ind_possible).*(1+ts_temp).*(1-4*tt(ind_possible).*ts_temp)))+calp*n/2*log((1+2*s*p*tt(ind_possible))./(1+2*s*p*tt(ind_possible).*(1+ts_temp).*(1-2*tt(ind_possible).*ts_temp)));
            [g2(s),indtemp]=min(g2_temp);
            T_star2(s)=ts_temp(indtemp)-ts_temp(indtemp)^2;
            ind_for_tt=ind_possible(indtemp);
            gbpart2=alp*(8+2*s*p*(1-4*tt(ind_for_tt)*ts_temp(indtemp))^2+32*s*p*tt(ind_for_tt)^2*(1+ts_temp(indtemp))^2)/(1+2*s*p*tt(ind_for_tt)*(1-4*tt(ind_for_tt)*ts_temp(indtemp))*(1+ts_temp(indtemp)))^2+calp*(4+2*s*p*(1-2*tt(ind_for_tt)*ts_temp(indtemp))^2+8*s*p*tt(ind_for_tt)^2*(1+ts_temp(indtemp))^2)/(1+2*s*p*tt(ind_for_tt)*(1-2*tt(ind_for_tt)*ts_temp(indtemp))*(1+ts_temp(indtemp)))^2;
            gb2(s)=n*s*p*tt(ind_for_tt)^2*gbpart2;
            temp_pe_withoutB=exp(g2(s))./sqrt(gb2(s)*2*pi)./T_star2(s);
            fianlPewithoutB(kaind,pp)=fianlPewithoutB(kaind,pp)+s/ka*double(cc(ka,s))*(temp_pe_withoutB);%s/ka*double(cc(ka,s)) is by the definition of the PUPE
            if fianlPewithoutB(kaind,pp)>=1
                fianlPewithoutB(kaind,pp)=1;
                break;
            end

        end
        save('AUMAC_cyclic_decoding_n38400_delay02_03092025.mat',"fianlPewithoutB","n","pdb","alp","ka1","logm",'-mat');
        if fianlPewithoutB(kaind,pp)>=1
            fianlPewithoutB(kaind,1:pp)=1;
            break;
        end
    end
  stpoint=find(fianlPewithoutB(kaind,:)<=target_pe_without_B,1);
end
toc;

