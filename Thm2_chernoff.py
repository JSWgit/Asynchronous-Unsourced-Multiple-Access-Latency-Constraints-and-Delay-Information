# this code is for AUMAC TIT paper, single antenna, analyzed with chernoff bound with Gallager's rho trick

import torch as tor
from typing import Tuple #used for phoenix #change tuple to Tuple
import matplotlib.pyplot as plt
import mpmath
import time
mpmath.mp.dps = 100 

def logcc_large(a,b):
    return  float(mpmath.loggamma(a + 1) - mpmath.loggamma(b + 1) - mpmath.loggamma(a - b + 1))

def cc(a,b):
    
    return tor.lgamma(a+1)-tor.lgamma(b+1)-tor.lgamma(a-b+1)

def err(rho, rho1, s, ka, snr,dm,n,my_device):
    m=128
    t=tor.arange(1e-4, min(1/(2*rho), 3),0.01, device=my_device) # type: ignore 
    t=tor.cat([tor.tensor([1e-10]),t]).to(my_device)
    t_u, bar_a=tor.meshgrid(t, tor.arange(s+1,device=my_device,dtype=tor.float), indexing='ij')
    p=10**(snr/10)
    theta=tor.tensor(logcc_large(2**m-int(ka),int(s))).to(my_device)
    theta0=(rho*1*theta+cc(ka-1,s)+tor.lgamma(s+1)).to(my_device)
    theta1=(rho*1*theta+cc(ka-1,s-1)+tor.lgamma(s+1)).to(my_device) 
    q=(1-rho)*(n-dm)/2*tor.log(1+2*s*p*t_u)+(1-rho)*(dm)/2*tor.log(1+2*bar_a*p*t_u)-n/2*tor.log(2*rho*t_u)
    coef1=(1-2*rho*t_u)/2
    As=(1+2*s*p*t_u*(1+rho))/(rho*t_u)
    A1=(1+2*p*t_u*(bar_a+rho))/(rho*t_u)
    A0=(1+2*p*t_u*(bar_a))/(rho*t_u)
    fa0=(n-dm)/2*tor.log(1+coef1*As)+dm/2*tor.log(1+coef1*A0)
    fa1=(n-dm)/2*tor.log(1+coef1*As)+dm/2*tor.log(1+coef1*A1)
    part0_val, part0_ind=tor.exp(theta0+1*q-1*fa0).min(dim=0)
    part1_val, part1_ind=tor.exp(theta1+1*q-1*fa1).min(dim=0)
    part0_val,_=part0_val[1:].max(dim=0)
    part1_val,_=part1_val.max(dim=0)
    for s1 in tor.arange(1,ka-s+1):
        barlam_0=(1+ coef1*A0)**(-1)*4*s1*p
        barlam_1=(1+ coef1*(1+2* rho*p*t_u)/(rho*t_u))**(-1)*4*s1*p
        theta_s1=1*(cc(ka-s,s1)+tor.lgamma(s+s1+1)-tor.lgamma(s+1))
        addterm_0=(1-2 *rho*t_u)*s1*p*(n-dm)*(1+coef1*As)**(-1)*(1+2*coef1*barlam_0+(2*coef1*barlam_0)**2/12 )**(-0.5)
        addterm_1=(1-2 *rho*t_u)*s1*p*(n-dm)*(1+coef1*As)**(-1)*(1+2*coef1*barlam_1+(2*coef1*barlam_1)**2/12 )**(-0.5)
        part0_s1_val, part0_s1_ind=tor.exp(theta0+theta_s1+1*q-1*(fa0+addterm_0)).min(dim=0)
        part1_s1_val, part1_s1_ind=tor.exp(theta1+theta_s1+1*q-1*(fa1+addterm_1)).min(dim=0)
        part0_s1_val,_=part0_s1_val[1:].max(dim=0)
        part1_s1_val,_=part1_s1_val.max(dim=0)
        part0_val+=part0_s1_val
        part1_val+=part1_s1_val

    err_final=(part0_val+part1_val)**(rho1)
    err_final_val, err_final_ind=err_final.min(dim=-1)
    return err_final_val

    
def main():
    my_device=tor.device("cuda" if tor.cuda.is_available() else "cpu")
    rho1=tor.arange(1e-2,1+1e-2,1e-2,device=my_device)
    rho1=tor.cat([tor.tensor([1e-10],device=my_device),rho1])
    alp=0.2
    n=tor.tensor(38400)
    ka_range=tor.arange(20,301,20)
    ka_range=tor.cat([tor.tensor([1]),tor.tensor([10]),ka_range]).to(my_device)
    snr_range=tor.arange(-24,-10,1e-2, device=my_device)
    dm=alp*n
    rho_range=tor.arange(1e-2,1+1e-2,1e-2, device=my_device)
    rho_range=tor.cat([tor.tensor([1e-10],device=my_device),rho_range])
    err_record=tor.zeros(len(ka_range), len(snr_range),device=my_device)
    start=0
    start_temp=0
    for i_ka, ka in enumerate(ka_range):
        start_cond=0
        start=start_temp
        print(f'[now we are in ] {ka}')
        s_range = tor.cat((
            ka[None],                                  # [ka]
            ka.new_tensor([1]),
            tor.arange(2, ka, device=my_device).flip(0) # type: ignore
        )) if ka > 2 else ka.new_tensor([1])
        for i_snr, snr in enumerate(snr_range[start:]):
            for s in s_range:
                err_temp=tor.tensor(100,device=my_device)
                for rho in rho_range:
                    err_temp1=err(rho, rho1, s, ka, snr,dm,n,my_device)
                    if err_temp1<=err_temp:
                        err_temp=err_temp1
                err_record[i_ka, start+i_snr]+=s/ka*err_temp
                print(f'[err, s, snr] are {err_record[i_ka, start+i_snr]}, {s}, {snr}')

                if err_record[i_ka, start+i_snr]>0.05:
                    break
            if err_record[i_ka, start+ i_snr]<1 and start_cond==0:
                start_cond=1
                start_temp=start+i_snr# type: ignore
            if err_record[i_ka, start+i_snr]<1e-10:
                break
    data = {
    'ka': ka_range.cpu(),
    'snr': snr_range.cpu(),
    'err':err_record.cpu(),
    'alp':alp,
    'n': n
    }
    # tor.save(data, 'my_data_AUMAC_Chernoff_thm2.pt')    
if __name__ == "__main__":
    main()
    


