# Import Library yang akan dipakai (JANGAN DI HAPUS, ISI DIBAWAH INI)
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image


classifier = joblib.load('model.pkl')

#scaler
scaler = joblib.load('scaler.pkl')
#df original
df = pd.read_csv('Adult census income dataset.csv')
df_preprocessed = pd.read_csv('Adult_census_income_preprocessed.csv')


def prediction(age,	workclass,	educationNum,	maritalStatus,	occupation,	relationship,	race,	sex,	hoursPerWeek,	capitalGain,capitalLoss):
    raw_data = {
        'age': age,
        'workclass': workclass,
        'educationNum': educationNum,
        'maritalStatus': maritalStatus,
        'occupation': occupation,
        'relationship': relationship,
        'race' :race,
        'sex':sex,
        'hoursPerWeek': hoursPerWeek,
        'capital':capitalGain-capitalLoss
    }

    clean_data = pipeline(raw_data)
    
    print(clean_data)
    pred = classifier.predict(clean_data)
    proba = classifier.predict_proba(clean_data)
    return pred,proba


def pipeline(data):
    #workclass
    workclass_order = {
        'Never-worked':0.000000,
        'Without-pay':0.000000,
        'Private':0.210093,
        'State-gov' : 0.271957,
        'Self-emp-not-inc' : 0.284927,
        'Local-gov' : 0.294792,
        'Federal-gov' : 0.386458,
        'Self-emp-inc' : 0.557348
    }

    data['workclass'] = workclass_order[data['workclass']]
    
    #education
    loweducation = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th']
    mideducation = ['HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters']
    higheducation = ['Doctorate','Prof-school']
    
    if data['educationNum'] in loweducation:
        data['educationNum'] = 0
    elif data['educationNum'] in mideducation:
        data['educationNum'] = 1
    elif data['educationNum'] in higheducation: 
        data['educationNum'] = 2


    #maritalstatus
    if data['maritalStatus'] in ['Married-civ-spouse', 'Married-AF-spouse'] : 
        data['maritalStatus'] = 1
    elif data['maritalStatus'] in ['Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Never-married'] : 
        data['maritalStatus'] = 0
    
    #occupation

    occupation_order = {
        'Priv-house-serv':0.006711,
        'Other-service':0.041578,
        'Handlers-cleaners':0.062774,
        'Armed-Forces':0.111111,
        'Farming-fishing':0.115694,
        'Machine-op-inspct':0.124875,
        'Adm-clerical':0.134483,
        'Transport-moving':0.200376,
        'Craft-repair':0.226641,
        'Sales':0.269315,
        'Tech-support':0.304957,
        'Protective-serv':0.325116,
        'Prof-specialty':0.342637,
        'Exec-managerial':0.484014
    }

    data['occupation'] = occupation_order[data['occupation']]

    #relationship
    
    if data['relationship'] in ['Wife', 'Husband']:
        data['relationship'] = 1
    elif data['relationship'] in ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative']:
        data['relationship'] = 0

    #race
    race_mapping = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
    data['race'] = race_mapping[data['race']]

    #sex
    sex_mapping = {'Female': 0, 'Male': 1}
    data['sex'] = sex_mapping[data['sex']]

    
    clean_data = []
    for key,value in data.items():
        clean_data.append(value)
    
    #scale clean_data
    clean_data = np.array(clean_data).reshape(1,-1)
    clean_data = scaler.transform(clean_data)

    return clean_data

# # Buat fungsi yang dapat mengeluarkan metrik evaluasi model (JANGAN DI HAPUS, ISI DIBAWAH INI)

def evaluate_model(X_test, y_test):
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_pred_proba)

    return acc,prec,rec,f1,roc_auc,y_pred,y_pred_proba

# # Buat fungsi untuk membuat visualisasi plot kurva ROC (JANGAN DI HAPUS, ISI DIBAWAH INI)

def plot_roc_curve(fpr,tpr,auc):
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label=f'AUC = {auc}',color = 'blue')
    plt.plot([0,1],[0,1],color = 'gray',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# # Buat fungsi untuk membuat visualisasi confusion matrix (JANGAN DI HAPUS, ISI DIBAWAH INI)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# # Buat fungsi utama yang akan memuat semua fungsi di atas dan ditampilkan pada Streamlit (JANGAN DI HAPUS, ISI DIBAWAH INI)

def main():
    menu_selection = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "Model Performance"],
        icons=["house-fill", "exclamation-triangle"],
        default_index=0,
        orientation="horizontal",
    )

    if menu_selection == "Home":
        st.markdown("<div align='center'>", unsafe_allow_html=True)
        st.markdown("<h1><strong>Tugas Besar Mata Kuliah Data Mining 2024</strong></h1>", unsafe_allow_html=True)
        st.markdown("<h2><strong>Kelompok 2 ✌🏻</strong></h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Team member table
        st.markdown("### Nama Anggota")
        st.table({
            "Nama Anggota": ["Alvaro Cleosanda", "Vilson", "Alisha Deanova Oemar", "Puja Daffa Adilah"],
            "NIM": ["1202220181", "1202220199", "1202223105", "1202223369"],
        })

        st.markdown("---")
        st.markdown("<div align='center'>", unsafe_allow_html=True)
        st.markdown("<h1><strong>💵 Employee Demographics and Income Prediction 💵</strong></h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
            Proyek ini mengeksplorasi **Dataset Demografi Karyawan dan Prediksi Pendapatan**, yang dirancang untuk menganalisis 
            berbagai faktor yang memengaruhi pendapatan, status kerja, dan karakteristik pekerjaan.
        """)

        st.markdown("## 📚 **Latar Belakang Masalah**")
        st.write("""
            Kesenjangan pendapatan dan faktor demografi memainkan peran penting dalam tenaga kerja. 
            Dengan meneliti karakteristik karyawan dan pendapatan, proyek ini bertujuan untuk 
            mengidentifikasi faktor-faktor kunci yang memengaruhi penghasilan, jenis pekerjaan, dan lainnya.
        """)

        st.markdown("## 📊 **Gambaran Umum Dataset**")
        st.write("""
            Dataset yang digunakan dalam proyek ini terdiri dari fitur-fitur terkait demografi karyawan dan karakteristik pekerjaan.
        """)
        st.write("""
            - **age**: Umur pekerja, direpresentasikan sebagai angka.
            - **workclass**: Kelas atau jenis pekerjaan (misalnya, pemerintahan, sektor swasta).
            - **fnlwgt**: Bobot akhir mewakili jumlah orang dengan karakteristik yang sama.
            - **education**: Tingkat pendidikan terakhir.
            - **education.num**: Tingkat pendidikan terakhir dalam angka.
            - **marital.status**: Status pernikahan.
            - **occupation**: Pekerjaan atau profesi spesifik.
            - **relationship**: Status hubungan dalam keluarga.
            - **race**: Etnis pekerja.
            - **sex**: Jenis kelamin pekerja.
            - **capital.gain**: Pendapatan dalam dolar.
            - **capital.loss**: Pengeluaran terkait pekerjaan.
            - **hours.per.week**: Jam kerja per minggu.
            - **native.country**: Negara asal pekerja.
            - **income**: Kategori pendapatan, baik ≤50K atau >50K.
        """)

        #dataset sample without first column
        st.markdown("## 📊 **Sample Dataset**")
        st.write(df.head(5))

        st.markdown("---")

    elif menu_selection == "Prediction":
        html_temp = """
        <div style = "background-color : darkblue;padding13px; border-radius:15px;margin-bottom:20px;">
        <h1 style = "color:white;text-align:center;">Income Prediction</h1>
        </div>
        """

        st.markdown(html_temp,unsafe_allow_html=True)

        age = st.number_input("Age",min_value=0,max_value=100)
        workclass = st.selectbox("Workclass",['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked'])
        
        educationNum = st.selectbox("Education",['HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate', 'Prof-school', 'Bachelors', 'Masters', '11th', 'Assoc-acdm', 'Assoc-voc', '1st-4th', '5th-6th', '12th', '9th', 'Preschool'])

        maritalStatus = st.selectbox("Marital Status", ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'] )

        occupation = st.selectbox("Occupation",['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty', 'Other-service', 'Adm-clerical', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners', 'Sales', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'] )

        relationship = st.selectbox("Relationship",['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])

        race = st.selectbox("Race",['White', 'Black', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'])

        sex = st.selectbox("Gender",['Female', 'Male'])
        
        hoursPerWeek = st.number_input("Hours of Work Per Week")

        capitalGain = st.number_input("Capital Gain")

        capitalLoss = st.number_input("Capital Loss")

        result = ""
        proba_result = ""

        if st.button("Predict"):
            result,proba = prediction(age,	workclass,	educationNum,	maritalStatus,	occupation,relationship,race,sex,hoursPerWeek,capitalGain,capitalLoss)
            result = '>50k' if result[0] == 1 else '<=50k'
            proba_result = f'{proba[0][0]}' if result == '<=50k' else f'{proba[0][1]}'
        
        st.success(f'Prediction : {result}')
        st.info(f'Confidence Score : {proba_result}')
    
    elif menu_selection == "Model Performance":
        html_temp = """
        <div style = "background-color : darkblue;padding13px; border-radius:15px;margin-bottom:20px;">
        <h1 style = "color:white;text-align:center;">Model Performance</h1>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        X = df_preprocessed.drop(['income'],axis=1)
        y = df_preprocessed['income']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        accuracy,precision,recall,f1,roc_auc,y_pred,y_pred_proba = evaluate_model(X_test,y_test)

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.success(f'Accuracy : {accuracy:.2f}%')
        with col2:
            st.info(f'Precision : {precision:.2f}%')
        with col3:
            st.warning(f'Recall : {recall:.2f}%')
        with col4:
            st.error(f'F1 Score : {f1:.2f}%')

        plot_option = st.selectbox('Choose Plot',['ROC AUC Curve','Confusion Matrix'])

        if plot_option == "ROC AUC Curve":
            fpr,tpr,_ = roc_curve(y_test,y_pred_proba)
            plot_roc_curve(fpr,tpr,roc_auc)
        elif plot_option == "Confusion Matrix":
            cm = confusion_matrix(y_test,y_pred)
            plot_confusion_matrix(cm)
   
if __name__=='__main__':
    main()