import glob
import dask.dataframe as dd
import dask
import pickle
import sys, os
import numpy as np
import time 
import pandas as pd

class vehicle:
    def __init__(self,id):
        self.ID=id
        self.states=[]
        self.frames=[]
        self.ls=[]
        self.frvs=[]
        self.flvs=[]
        self.rrvs=[]
        self.rlvs=[]
        self.rvs=[]
        self.lvs=[]
        self.llvs=[]

    def __repr__(self):
        return 'vehicle obj:'+str(self.ID)

    def populate_veh_lists(self,cluster):
        self.frvs.append(cluster['fr'])
        self.flvs.append(cluster['fl'])
        self.rrvs.append(cluster['rr'])
        self.rlvs.append(cluster['rl'])
        self.rvs.append(cluster['r'])
        self.lvs.append(cluster['l'])

    def bin_side_vehs(self,rvs,lvs):
        cluster={'rl':[0,0,0,0,0,0],'rr':[0,0,0,0,0,0],'r':[0,0,0,0,0,0],'fl':[0,0,0,0,0,0],'fr':[0,0,0,0,0,0],'l':[0,0,0,0,0,0]}
        if self.ls[-1][0]==0:
            self.populate_veh_lists(cluster)
            return

        lvState=self.ls[-1]
        # select neighbors in left lane 
        for vs in lvs:
            veh=vs[1:]-self.states[-1][1:]
            if vs[0]==0:
                continue
            if veh[1]<lvState[2]+0.16 and veh[1]>lvState[2]-16:
                cluster['l']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
            elif veh[1]<lvState[2]-16:
                if np.sum(cluster['rl']) == 0:
                    cluster['rl']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])  
                elif cluster['rl'][2]<veh[1]:
                    cluster['rl']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])

            elif veh[1]>lvState[2]+0.16:
                if np.sum(cluster['fl']) == 0:
                    cluster['fl']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
                elif cluster['fl'][2]>veh[1]:
                    cluster['fl']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])

        # select neighbors in right lane 
        for vs in rvs:
            if vs[0]==0:
                continue
            veh=vs[1:]-self.states[-1][1:]
            if veh[1]<lvState[2]+0.16 and veh[1]>lvState[2]-16:
                cluster['r']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
            elif veh[1]+16<lvState[2]:
                if np.sum(cluster['rr'])==0:
                    cluster['rr']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
                elif cluster['rr'][2]<veh[1]:
                    cluster['rr']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])

            elif veh[1]>lvState[2]+0.16:
                if np.sum(cluster['fr'])==0:
                    cluster['fr']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
                elif cluster['fr'][2]>veh[1]:
                    cluster['fr']=np.append([vs[0]],veh,[(vs[3]-lvState[2])/(vs[2]-lvState[1])])
        
        self.populate_veh_lists(cluster)
        # if np.sum(cluster['l']) or np.sum(cluster['r']): 
        #     print(self.ID)
        #     print(cluster)
        #     print('####')


files=glob.glob('*.csv')
for f in files:
    dfs=pd.read_csv(f)
    os.mkdir(f[:-4])
    os.chdir(f[:-4])

    # for row_data in dfs.iterrows():
    #   print(type(dfs[dfs['a']==3]['b'].compute().values))

        # frame=row_data[1]['Frame ID']
        # vehs=row_data[1][]

    vehicleObjs={}
    frame=0
    state_feats=['Vehicle_ID','Local_X','Local_Y','v_Vel','v_Acc']
    for i in range(8702):
        frame+=1
        print('### '+str(frame)+ ' ###')
        st_time=time.time()
        # Extract all vehs in frame
        dff=dfs[dfs['Frame_ID']==frame]

        print('Table extracted: '+str(time.time()-st_time))
        if len(dff)==0:
            continue

        v_states={'0':[0,0,0,0,0]}
        vids=dff['Vehicle_ID']

        st_time1=time.time()
        for v in vids:
            if not str(v) in vehicleObjs:
                vehicleObjs[str(v)]=vehicle(v)
            v_states[str(v)]=dff[dff['Vehicle_ID']==v][state_feats].to_numpy().squeeze()
            
        
        st_time2=time.time()
        print('states created: '+str(st_time2-st_time1))
        for v in vids:
            lvid=dff[dff['Vehicle_ID']==v]['Preceeding'].to_numpy().squeeze()
            llvid=dff[dff['Vehicle_ID']==lvid]['Preceeding'].to_numpy().squeeze()
            if str(lvid) in v_states and str(llvid) in v_states:
                vehicleObjs[str(v)].frames.append(frame)
                vehicleObjs[str(v)].states.append(v_states[str(v)])
                lane=dff[dff['Vehicle_ID']==v]['Lane_ID'].to_numpy().squeeze()
                lvs=dff[dff['Lane_ID']==lane-1][state_feats].to_numpy()
                rvs=dff[dff['Lane_ID']==lane+1][state_feats].to_numpy()
                if llvid>0 and lvid>0:
                    rel_state=v_states[str(llvid)]-v_states[str(lvid)]+np.array([lvid,0,0,0,0])
                    vehicleObjs[str(v)].llvs.append(rel_state)
                elif lvid>0:
                    vehicleObjs[str(v)].llvs.append(v_states[str(llvid)])
                vehicleObjs[str(v)].ls.append(v_states[str(lvid)],np.append([-1*vehicleObjs[str(v)].llvs[-1][3]/vehicleObjs[str(v)].llvs[-1][2]]))
                vehicleObjs[str(v)].bin_side_vehs(rvs,lvs)
        print('scene processed: '+str(time.time()-st_time2))
    print('saving veh objs')
    for k,v in vehicleObjs.items():
        print(k+' done')
        pickle.dump(v,open(k+'.vehicleObj','wb'))

    os.chdir('..')




