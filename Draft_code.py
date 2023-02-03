import pandas as pd
import random
import datetime
import numpy as np

def generate_data(subject_id):
    data = []
    # Assign a random number of visits (up to 7, including EOT)
    num_visits = random.randint(1, 7)
    visits = ["Screening"] + ["{}-week".format(i) for i in range(4, 4*num_visits, 4)]
    #Generating date associated to Visit
    date_screening = datetime.date(2018, 1, 1) - datetime.timedelta(days=random.randint(0,365))
    dates= [date_screening + pd.Timedelta(weeks=i*4) + pd.Timedelta(days=random.randint(-5,5)) for i in range(num_visits)]
    # Assign a random number of target lesions (up to 5)
    num_target_lesions = random.randint(1, 5)
    # Assign a random number of non-target lesions (up to 10)
    num_non_target_lesions = random.randint(1, 10) if random.random() > 0.9 else random.randint(1, 5)
    # Assign a random number of new lesions (up to 10, but very unlikely to have more than 3)
    num_new_lesions = 0 # Start with no new lesions at screening
    # Assign a random visit as the end of treatment (EOT)
    eot_visit = random.choice(visits) if random.random() > 0.5 else None
    # Assign target lesion IDs
    target_lesion_ids = ["T{}".format(i) for i in range(1, num_target_lesions + 1)]
    # Assign non-target lesion IDs
    non_target_lesion_ids = ["NT{}".format(i) for i in range(1, num_non_target_lesions + 1)]
    # Assign new lesion IDs
    new_lesion_ids = ["NL{}".format(i) for i in range(1, num_new_lesions + 1)]
    # List of locations
    locations = ["Liver", "Lung", "Brain",'Liver','Liver', 'Lung', 'Brain', 'Kidney'
                , 'Lymph Nodes', 'Lung', 'Lung', 'Lymph Nodes', 'Lymph Nodes',
                'Lung', 'Lung','Lung', 'Lung','Adrenal Gland']
    locations_nt = ['Lung', 'Lung','Lung', 'Lung','Adrenal Gland', "Liver", "Lung", 'Brain', 'Kidney'
                    'Lung', 'Lymph Nodes', 'Lymph Nodes', 'Lymph Nodes', 
                    'Lung',"Brain",'Liver','Liver', 'Lung']
    
    random.shuffle(locations)
    random.shuffle(locations_nt)
    
    for visit, date in zip(visits, dates):
        if visit != "Screening":
            # Increase the number of new lesions in follow-up visits (up to 10)
            num_new_lesions = random.randint(0, 10) if random.random() > 0.9 else random.randint(0, 3)
            new_lesion_ids = ["NL{}".format(i) for i in range(1, num_new_lesions + 1)]
        # Assign target lesion measurements
        for i, lesion_id in enumerate(target_lesion_ids):
            #random.shuffle(locations)
            location = locations[i%len(locations)] # Assign a different location for each lesion
            measurement = random.normalvariate(50, 10)
            data.append([subject_id, visit, date, "Target", lesion_id, location, measurement])
        # Assign non-target lesion measurements
        for i, lesion_id in enumerate(non_target_lesion_ids):
            location = locations_nt[i%len(locations_nt)] # Assign a different location for each lesion
            measurement =""
            data.append([subject_id, visit, date, "Non-Target", lesion_id, location, measurement])
    
    df = pd.DataFrame(data, columns=["Subject ID", "Visit", "date", "Method", "Lesion ID", "Location", "Measurement"])
    df_tg=df[df['Method']=="Target"]
    df_tg.loc[:,'SoD']=df_tg.groupby(['Subject ID','Visit'])['Measurement'].transform('sum')
    df_tg.loc[:,'SoD_Exc_LN']=df_tg.loc[df_tg['Location'] != 'Lymph Nodes', :].groupby(['Subject ID', 'Visit'])['Measurement'].transform('sum')

    df_tg.loc[:,'SoD_BL']=min(df_tg[df_tg['Visit']=='Screening']['SoD'])
    #Calculate Nadir
    def min_val(row):
        mask = df_tg[(df_tg['date'] <= row['date'])]
        min_SoD = mask[mask['Subject ID']==row['Subject ID']]['SoD'].min()
        return min_SoD
    df_tg.loc[:, 'SoD_Nadir'] = df_tg.apply(min_val, axis=1)
    df_tg.loc[:, 'PERC_NADIR']= (df_tg.loc[:, 'SoD']/df_tg.loc[:, 'SoD_Nadir'])-1
    df_tg.loc[:, 'PERC_BL']= (df_tg.loc[:, 'SoD']/df_tg.loc[:, 'SoD_BL'])-1
    df_tg.loc[:, 'NADIR_DIFF']= df_tg.loc[:, 'SoD']-df_tg.loc[:, 'SoD_Nadir']
    df_tg.loc[:,'Number_LN'] = df_tg.groupby('Visit')['Location'].transform(lambda x: x[x == 'Lymph Nodes'].count())
    df_tg.loc[:,'Number_LN_10'] = df_tg.loc[df_tg['Location'] == 'Lymph Nodes', :].groupby('Visit')['Measurement'].transform(lambda x: (x >= 10).sum())
    df_tg.loc[:,'Number_LN_10'] = df_tg['Number_LN_10'].where(df_tg['Location'] == 'Lymph Nodes', 0)

    df_tg.loc[:, 'R_flag'] = np.where(
    df_tg.loc[:, 'SoD'].isnull(),
    'NA',
        np.where(
            (df_tg.loc[:, 'NADIR_DIFF'] >= 5) & (df_tg.loc[:, 'PERC_NADIR'] >= 0.2),
            'PD',
            np.where(
                (df_tg.loc[:, 'Number_LN_10'].isnull()) &
                ((df_tg.loc[:, 'SoD_Exc_LN'] == 0) | df_tg.loc[:, 'SoD_Exc_LN'].isnull()),
                'CR',
                np.where(
                    df_tg.loc[:, 'PERC_BL'] <= -0.3, 'PR', 'SD'))))
    
    # Final Table
    df = pd.merge(df, df_tg, on =['Subject ID','Visit','date','Method','Lesion ID', 'Location', 'Measurement'], how ='left')
    #df = df[['Subject ID', 'Visit', 'date', 'Method', 'Lesion ID', 'Location', 'Measurement', 'SoD', 'SoD_BL', 'SoD_Nadir', 'PERC_NADIR', 'PERC_BL', 'NADIR_DIFF', 'R_flag']]
    pd.set_option('display.float_format', '{:.2f}'.format)
    return df


generate_data('123')
