clear;clc;
syms cc(w,m)
cc(w,m) = nchoosek(w,m);
load("AUMAC_cyclic_decoding_n38400_delay02_03092025.mat");
calp=1-alp;
thr=100;
stepsize=1e-3;
target_pe_without_B=1e-14;
tt=[0.255:1e-2:20];
ts=zeros(length(tt),1/stepsize+1);
for z=1:length(tt)
    t=tt(z);
    ts(z,1:length(0:stepsize:1/4/t))=0:stepsize:1/4/t;
end
fianlPewithoutB(16,1:39)=1;
power_withB=[0;0];
step_size_for_B=4e-2;

for kaind=1:length(ka1)
    % if ka1(kaind)>=110
        temp=pdb(find(fianlPewithoutB(kaind,:)<1,1)):step_size_for_B:pdb(find(fianlPewithoutB(kaind,:)<1,1))+3.6;
    % else
    %     temp=pdb(find(fianlPewithoutB(kaind,:)<1,1)+1)+0.2:step_size_for_B:pdb(find(fianlPewithoutB(kaind,:)<1,1)+1)+1.2;
    % end
        temp2=[temp;ones(1,length(temp))*ka1(kaind)];
        power_withB=[power_withB,temp2];
end
power_withB(:,1)=[];
PewithonlyB=zeros(1,length(power_withB(1,:)));
PewithoutB=zeros(1,length(power_withB(1,:)));
PewithonlyB2=zeros(1,length(power_withB(1,:)));
PewithonlyB33=zeros(1,length(power_withB(1,:)));

binomcoef=zeros(max(ka1),max(ka1),length(ka1));
for kaz=1:length(ka1)
    ka=ka1(kaz);
    for zzz=1:ka-1
        binomcoef(zzz,1:ka-zzz,kaz)=double(cc(ka-zzz,[1:ka-zzz]));
        % zs
    end
end

% zzz=87;tic;
%%
 for zzz=1:length(power_withB(1,:))
% zzz=481;
    p=10^(power_withB(1,zzz)/10);
    ka=power_withB(2,zzz);
    g2=thr*ones(1,ka);
    T_star2=zeros(1,ka);
    gb2=zeros(1,ka);
    PewithonlyB_before_multiply_binomial=zeros(ka,ka);
    PewithonlyB_before_multiply_binomial2=zeros(ka,ka);
    PewithonlyB_before_multiply_binomial33=zeros(ka,ka);
    for s=1:ka
        g_bar=thr*ones(ka,ka);
        T_star=ones(ka,ka);gb=ones(1,ka-s);
        logthe=log(cc(exp(logm)-ka,s))+s*log(1+alp*n);
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
        %%
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
        PewithoutB(zzz)=PewithoutB(zzz)+s/ka*double(cc(ka,s))*(temp_pe_withoutB);%s/ka*double(cc(ka,s)) is by the definition of the PUPE
        if PewithoutB(zzz)>=1
            PewithoutB(zzz)=1;
            break;
        end
        %%
        %the pe with B
        %for the case with B, there are s1 mismatch delay terms for the correctly decoded codewords.
        CB=double(-logthe+n/2*log(1+2*s*p*tt));
        f1=(1-2*tt'.*ts).*tt'.*ts./(1+2*s*p*tt'.*(1+ts).*(1-2*tt'.*ts));
        f2=(1-4*tt'.*ts).*tt'.*ts./(1+2*s*p*tt'.*(1+ts).*(1-4*tt'.*ts));
        f1d=tt'.*(1-4*tt'.*ts+2*s*p*tt'.*(1-2*tt'.*ts).^2)./(1+2*s*p*tt'.*(1+ts).*(1-2*tt'.*ts)).^2;
        f2d=tt'.*(1-8*tt'.*ts+2*s*p*tt'.*(1-4*tt'.*ts).^2)./(1+2*s*p*tt'.*(1+ts).*(1-4*tt'.*ts)).^2;
        f1dd=-[4*tt'.^2.*(1+2*s*p*tt'+3*s^2*p^2*tt'.^2.*ts.^2.*(1-3*tt'.*ts).^2+2*s^2*p^2*tt'.*(1-2*tt'.*ts).^3)]./(1+2*s*p*tt'.*(1+ts).*(1-2*tt'.*ts)).^3;
        f2dd=-[4*tt'.^2.*(2+4*s*p*tt'+12*s^2*p^2*tt'.^2.*ts.^2.*(1-6*tt'.*ts).^2+2*s^2*p^2*tt'.*(1-4*tt'.*ts).^3)]./(1+2*s*p*tt'.*(1+ts).*(1-4*tt'.*ts)).^3;

        bias=-CB'+alp*n*s*p*tt'.*(8*tt'.*ts+4*tt'-1)./(1+2*s*p*tt'.*(1+ts).*(1-4*tt'.*ts))+calp*n*s*p*tt'.*(4*tt'.*ts+2*tt'-1)./(1+2*s*p*tt'.*(1+ts).*(1-2*tt'.*ts));
        for s1=1:ka-s
            gammau=2*(s1)*p*n*(alp*f2d+calp*f1d);
            gammau(f2d<0)=4*n*(s1)*p*f1d(f2d<0);
            biasu=bias-gammau;
            ind_possible_upper2=find(biasu(:,2)<=0);
            remove_term= sum(biasu(ind_possible_upper2,:)>0,2)>0; % to remove those t can't achieve 0
            ind_possible_upper=ind_possible_upper2(remove_term);
            biasu_used=biasu(ind_possible_upper,:);
            lengthofpossibleterms_upper=length(ind_possible_upper);
            ts_ind_lower=zeros(1,lengthofpossibleterms_upper);
            ts_ind_upper=zeros(1,lengthofpossibleterms_upper);
            if lengthofpossibleterms_upper==0
                PewithonlyB_before_multiply_binomial(s,s1)=1;
                continue
            end
            gammal=2*(s1)*p*n*(alp*f2d+calp*f1d)./(1+8*(s1)*p*f1);
            gammal(f2d<0)=4*(s1)*p*n*f2d(f2d<0);
            biasl=bias-gammal;
            biasl_used=biasl(ind_possible_upper,:); % the reason that here we use ind_possible_upper is because both the upper and lower bound should have the same "t".
            for z=1:lengthofpossibleterms_upper
                ts_ind_upper(z)=find(biasu_used(z,ts(ind_possible_upper(z),:)~=0)>=0,1)+1;
                ts_ind_lower(z)=find(biasl_used(z,ts(ind_possible_upper(z),:)~=0)>=0,1)+1;
            end

            tsl=ts(1,ts_ind_lower);
            tsu=ts(1,ts_ind_upper);
            barlambda=4*(s1)*p*diag(f1(ind_possible_upper,ts_ind_lower))'; % to extract the terms at ind_possible_upper rows and at ts_ind_lower columns, we use diag to extract these terms.
            g=-tsl.*CB(ind_possible_upper)+alp*n/2*log((1+2*s*p*tt(ind_possible_upper))./(1+2*s*p*tt(ind_possible_upper).*(1+tsl).*(1-4*tt(ind_possible_upper).*tsl)))+calp*n/2*log((1+2*s*p*tt(ind_possible_upper))./(1+2*s*p*tt(ind_possible_upper).*(1+tsl).*(1-2*tt(ind_possible_upper).*tsl)))-2*n*(s1)*p*(alp*diag(f2(ind_possible_upper,ts_ind_lower))+calp*diag(f1(ind_possible_upper,ts_ind_lower)))'./sqrt(1+2*barlambda+barlambda.^2/3);
            T_star_vector=min( [tsl- tsl.^2;tsu-tsu.^2]);
            gbpart12=alp*(8+2*s*p*(1-4*tt(ind_possible_upper)'.*ts(ind_possible_upper,:)).^2+32*s*p*tt(ind_possible_upper)'.^2.*(1+ts(ind_possible_upper,:)).^2)./(1+2*s*p*tt(ind_possible_upper)'.*(1-4*tt(ind_possible_upper)'.*ts(ind_possible_upper,:)).*(1+ts(ind_possible_upper,:))).^2+calp*(4+2*s*p*(1-2*tt(ind_possible_upper)'.*ts(ind_possible_upper,:)).^2+8*s*p*tt(ind_possible_upper)'.^2.*(1+ts(ind_possible_upper,:)).^2)./(1+2*s*p*tt(ind_possible_upper)'.*(1-2*tt(ind_possible_upper)'.*ts(ind_possible_upper,:)).*(1+ts(ind_possible_upper,:))).^2;
            gbpart1=n*s*p*tt(ind_possible_upper)'.^2.*gbpart12;
            gbpart2=min(gbpart1');
            gbpart3=n*s*p*tt(ind_possible_upper)'.^2.*gbpart12-2*s1*n*p*(calp*f1dd(ind_possible_upper,:)+alp*f2dd(ind_possible_upper,:))./(1+8*s1*p*f1(ind_possible_upper,:)).*(ts(ind_possible_upper,:)>0);
            for zs=1:length(ind_possible_upper)
                gbpart33(zs)=min(gbpart3(zs,ts(ind_possible_upper(zs),:)>0));
            end
            Pe_before_min_t_ts33=exp(g+s1*log(1+alp*n))./T_star_vector./sqrt(2*pi*gbpart33(1:length(ind_possible_upper)));
            PewithonlyB_before_multiply_binomial33(s,s1)=(min(Pe_before_min_t_ts33));
            if PewithonlyB_before_multiply_binomial33(s,s1)<=1e-450
                break;
            end
        end
        temp_pe_withB33=sum(binomcoef(s,1:ka-s,ka1==ka).*PewithonlyB_before_multiply_binomial33(s,1:ka-s));
        PewithonlyB33(zzz)=PewithonlyB33(zzz)+s/ka*double(cc(ka,s))*(temp_pe_withB33);

        if PewithonlyB33(zzz)>=1
            PewithonlyB33(zzz)=1;
            break;
        end
        %%
    end
    save('AUMAC_cyclic_decoding_n38400_with_B_delay02_without_delay_information_03092025.mat',"PewithonlyB33","PewithoutB","power_withB","n","pdb","alp","ka1",'-mat');
 end
