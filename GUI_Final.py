#IMPORT LIBRARIES
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import squarify
from io import BytesIO
from datetime import datetime
from underthesea import word_tokenize, pos_tag, sent_tokenize
import jieba
import re
import string
from wordcloud import WordCloud
import pickle


STOP_WORD_FILE = 'Data/stopwords-en.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')

# USING MENU
st.title("Customer Segmentation")
menu = ["Trang chủ", "Tổng quan", "Cơ chế phân nhóm", "Phân tích khách hàng", "Đề xuất chiến lược"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(menu)
with tab1:
    st.image('Data/Featured-image-2.webp')
    st.write("""Bạn là chủ Cửa hàng bán lẻ. Bạn muốn phân tích tệp khách hàng?
             Ứng dụng này sẽ giúp cửa hàng bạn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.""")    
    if st.button("Let's try :point_right:"):
        st.write('Hãy thao tác theo các Tab tiếp theo :ok_hand:')
    st.write('#### Author 1: Triệu Thị Kim Trang')
    st.write('#### Author 2: Phan Thị Tuyết')
with tab2:
    st.image('Data/Business Overview.jpg')
    st.write('#### Vui lòng upload các file giao dịch và sản phẩm của cửa hàng')
    uploaded_trans = st.file_uploader("Upload Transactions file", type=['txt', 'csv'])
    uploaded_product= st.file_uploader("Upload Products file", type=['txt', 'csv'])
# LOADING DATA
    if uploaded_trans is not None and uploaded_product is not None:
        df_product = pd.read_csv(uploaded_product)
        df = pd.read_csv(uploaded_trans)
    else:
        df_product = pd.read_csv('Data/Products_with_Prices.csv')
        df = pd.read_csv('Data/Transactions.csv')
        
    df = df.merge(df_product, how='left', on='productId')
    df['Sales'] = df['items'] * df['price']
    df['Transaction_id'] = df.index
    string_to_date = lambda x : datetime.strptime(x, "%d-%m-%Y").date()

    # Convert InvoiceDate from object to datetime format
    df['Date'] = df['Date'].apply(string_to_date)
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df['Year'] = df['Date'].dt.year
    df['Year'] = df['Year'].astype(str)

    # Drop NA values
    df = df.dropna()
    # RFM
    # Convert string to date, get max date of dataframe
    max_date = df['Date'].max().date()

    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: len(x.unique())
    Monetary = lambda x : round(sum(x), 2)

    df_RFM = df.groupby('Member_number').agg({'Date': Recency,
                                            'Transaction_id': Frequency,
                                            'Sales': Monetary })
    # Rename the columns of DataFrame
    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
    # Descending Sorting
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)

    # Đọc model đã chọn lên
    pkl_filename = "cus_segment_model.pkl"  
    with open(pkl_filename, 'rb') as file:  
        cus_segment_model = pickle.load(file)
    cus_segment_model.fit(df_RFM)
    df_RFM["Cluster"] = cus_segment_model.labels_
    conditions = [
        (df_RFM['Cluster'] == 0),
        (df_RFM['Cluster'] == 1),
        (df_RFM['Cluster'] == 2),
        (df_RFM['Cluster'] == 3),
        (df_RFM['Cluster'] == 4)
    ]
    choices = ['REGULARS', 'NEW', 'VIP', 'LOYAL', 'LOST']
    df_RFM['CustGroup'] = np.select(conditions, choices, default='UNKNOWN')

    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg = df_RFM.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

    # Reset the index
    rfm_agg = rfm_agg.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    conditions2 = [
        (rfm_agg['Cluster'] == 0),
        (rfm_agg['Cluster'] == 1),
        (rfm_agg['Cluster'] == 2),
        (rfm_agg['Cluster'] == 3),
        (rfm_agg['Cluster'] == 4)
    ]
    choices2 = ['REGULARS', 'NEW', 'VIP', 'LOYAL', 'LOST']
    rfm_agg['Cluster'] = np.select(conditions2, choices2, default='UNKNOWN')
    # rfm_agg['Cluster'] = 'Cluster '+ rfm_agg['Cluster'].astype('str')
    

    # Reset the index
    df_RFM = df_RFM.reset_index()
    df_RFM.head()
    df_cust = df.merge(df_RFM[['Member_number','CustGroup']], on='Member_number', how='left')

    rfm = df_RFM[['Member_number','Recency','Frequency','Monetary']]

    df_plot = df.groupby('Year').agg({'Member_number': lambda x: len(x.unique()),
                                                'Transaction_id': lambda x: len(x.unique())
                                                }).reset_index()
    df_plot.columns = ['Năm', 'Số lượng KH', 'Số lượng đơn hàng']

    df_plot.sort_values(by = 'Số lượng KH', ascending = False, inplace = True)
    df_plot.reset_index(drop=True, inplace=True)     
    
    st.subheader("TỔNG QUAN TÌNH HÌNH KINH DOANH")
    st.write("##### 1. Tổng quan giao dịch mua hàng:")
    st.write("###### 1.a Đơn hàng:")
    # Đếm số lượng đơn hàng
    a = len(df['Transaction_id'].unique())
    b = len(df.loc[df['items'] >= 3]['Transaction_id'].unique())
    c = len(df.loc[df['Sales'] >= 50]['Transaction_id'].unique())
    st.dataframe(pd.DataFrame({'Đơn hàng': ['Tổng', 'Số lượng mua trên 3 cái/lần','Giá trị trên 50 USD/ lần'], 'Số lượng': [a,b,c]}))
    
    st.write("###### 1.b Giá trị đơn hàng cao nhất:")
    st.dataframe(df.loc[df['Sales'].idxmax()].to_frame().T)
    
    st.write("###### 1.c Thời gian thu thập dữ liệu:")
    start_date = df['Date'].min().strftime("%d/%m/%Y")
    end_date = df['Date'].max().strftime("%d/%m/%Y")
    st.write(f"Các thông tin mua hàng đang được ghi nhận từ {start_date} đến {end_date}")
    
    # Vẽ biểu đồ Số lượng đơn hàng theo năm
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(df_plot['Năm'], df_plot['Số lượng đơn hàng'], color='orange')
    plt.xlabel('Năm')
    plt.ylabel('Số lượng đơn hàng')
    plt.title('Thống kê số lượng đơn hàng theo năm')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}', ha='center', va='bottom')
    st.pyplot(plt)
    
    st.write("##### 2. Tổng quan sản phẩm:")
    # Thống kê số lượng KH và đơn hàng theo sản phẩm
    df_g_product = df.groupby('productName').agg({'Member_number': lambda x: len(x.unique()), 'Transaction_id': lambda x: len(x.unique())}).reset_index()
    df_g_product.columns = ['Sản phẩm', 'Số lượng KH', 'Số lượng đơn hàng']
    df_g_product.sort_values(by = 'Số lượng KH', ascending = False, inplace = True)
    df_g_product.reset_index(drop=True, inplace=True)
    df_bar = df_g_product.head()
    ## Top 5 SP theo số lượng khách hàng
    colors = ['salmon', 'limegreen','gold', 'pink','skyblue']
    sorted_indices = sorted(range(len(df_bar['Số lượng KH'])), key=lambda i: df_bar['Số lượng KH'][i], reverse=False)
    sorted_countries = [df_bar['Sản phẩm'][i] for i in sorted_indices]
    sorted_num_countries = [df_bar['Số lượng KH'][i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries, sorted_num_countries, color=colors)
    plt.xlabel('Số lượng KH')
    plt.ylabel('Sản phẩm')
    plt.title('Top 5 sản phẩm có nhiều khách hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)
    ## Top 5 SP theo số lượng đơn hàng
    df_g_product1 = df.groupby('productName').agg({'Member_number': lambda x: len(x.unique()), 'Transaction_id': lambda x: len(x.unique())}).reset_index()
    df_g_product1.columns = ['Sản phẩm', 'Số lượng KH', 'Số lượng đơn hàng']
    df_g_product1.sort_values(by = 'Số lượng đơn hàng', ascending = False, inplace = True)
    df_g_product1.reset_index(drop=True, inplace=True)
    df_bar1 = df_g_product1.head()
    sorted_indices1 = sorted(range(len(df_bar1['Số lượng đơn hàng'])), key=lambda i: df_bar1['Số lượng đơn hàng'][i], reverse=False)
    sorted_countries1 = [df_bar1['Sản phẩm'][i] for i in sorted_indices1]
    sorted_num_countries1 = [df_bar1['Số lượng đơn hàng'][i] for i in sorted_indices1]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries1, sorted_num_countries1, color=colors)
    plt.xlabel('Số lượng đơn hàng')
    plt.ylabel('Sản phẩm')
    plt.title('Top 5 sản phẩm có nhiều đơn hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)
      
    st.write("##### 3. Tổng quan lượng khách hàng:")
    # Vẽ biểu đồ số lượng KH theo năm
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df_plot['Năm'], df_plot['Số lượng KH'], color='skyblue')
    plt.xlabel('Năm')
    plt.ylabel('Số lượng KH')
    plt.title('Thống kê số lượng khách hàng theo năm')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}', ha='center', va='bottom')
    st.pyplot(plt)
    
    st.write("##### 4. Tổng quan một số cụm khách hàng:")

    def top_product(cust_group):
        df_top_cluster = pd.DataFrame(
            df_cust.loc[df_cust['CustGroup']== cust_group]
            .groupby(['productName','price']).count()['productId']
            .sort_values(ascending=False).head(20)).reset_index()
        return df_top_cluster
    def text_underthesea(text):
        products_wt = text.str.lower().apply(lambda x: word_tokenize(x, format="text"))
        products_name_pre = [[text for text in set(x.split())] for x in products_wt]
        products_name_pre = [[re.sub('[0-9]+','', e) for e in text] for text in products_name_pre]
        products_name_pre = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml']]
                            for text in products_name_pre] # ký tự đặc biệt
        products_name_pre = [[t for t in text if not t in stop_words] for text in products_name_pre] # stopword
        return products_name_pre
    def wcloud_visualize(input_text):
        flat_text = [word for sublist in input_text for word in sublist]
        text = ' '.join(flat_text)
        wc = WordCloud(
                        background_color='white',
                        colormap="ocean_r",
                        max_words=50,
                        width=1600,
                        height=900,
                        max_font_size=400)
        wc.generate(text)
        plt.figure(figsize=(8,12))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    user_selects = ['VIP','LOYAL','NEW','REGULARS', 'LOST']
    for name in user_selects:
        st.write("#### WordCloud top sản phẩm của PK KH", name)
        top_products = top_product(name)
        sample_text = top_products['productName']
        processed_text = text_underthesea(sample_text)
        wcloud_visualize(processed_text)
with tab3:
    st.image('Data/customer-segmentation-social.png')
    st.write('### Model RFM sử dụng thuật toán KMeans')
    st.write("""Customer Segmentation là một công cụ mạnh mẽ giúp doanh nghiệp hiểu sâu hơn về khách hàng của họ và cách tùy chỉnh chiến lược tiếp thị.
                            Đây là một bước không thể thiếu để đảm bảo rằng bạn đang tiếp cận và phục vụ mọi nhóm khách hàng một cách hiệu quả""")
    st.write("""**Các phân cụm khách hàng theo thuật toán KMeans:**""")
    st.write("""
            + KH VIP (VIP): Khách hàng có lượng chi tiêu lớn, tần suất tiêu thụ thường xuyên, và vừa mua hàng gần đây
            + KH MỚI (NEW): Khách hàng mới đến gần đây, chưa quan tâm đến mức độ chi tiêu
            + KH THÂN THIẾT (LOYAL): Khách hàng thường đến và vẫn còn đến gần đây
            + KH RỜI BỎ (LOST): Khách hàng quá lâu chưa đến
            + KH THÔNG THƯỜNG (REGULARS): Nhóm còn lại, thường ở mức trung bình ở 3 khía cạnh 'Recency', 'Frequency', 'Monetary'
            """)
    st.write('### Giá trị trung bình Recency-Frequency-Monetary theo các phân cụm')
    st.dataframe(rfm_agg)
    # Show biểu đồ phân cụm
    plt.figure()
    st.write('### TreeMap')
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)
    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
               'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}
    squarify.plot(sizes=rfm_agg['Count'],
				  text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                  color=colors_dict2.values(),
				  label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
						  for i in range(0, len(rfm_agg))], alpha=0.5 )
    plt.title("Customers Segments - Treemap",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(plt)
    
    st.write('### Scatter Plot (RFM)')
    fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
           hover_name="Cluster", size_max=100)
    fig.update_layout(template="plotly_dark")
    fig.update_layout(plot_bgcolor="white")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig)

with tab4:
    st.image('Data/segmentation.webp')
    st.subheader("PHÂN TÍCH KHÁCH HÀNG")
    type = st.radio("### Nhập thông tin khách hàng", options=["Xem KH hiện hữu", "Dự đoán KH mới"])
    if type == "Xem KH hiện hữu":
        st.subheader("Mã khách hàng")
        # Tạo điều khiển để người dùng nhập và chọn nhiều mã khách hàng từ danh sách gợi ý
        st.markdown("**Có thể nhập và chọn nhiều mã khách hàng từ danh sách gợi ý**")

        all_ids = df['Member_number'].unique()
        # Chọn nhiều ID từ danh sách
        selected_ids = st.multiselect("Chọn Member_number:", all_ids)
        # In ra danh sách ID đã chọn
        st.write("#### Bạn đã chọn các KH sau:")
        st.write(selected_ids)

        if any(id in df['Member_number'].values for id in selected_ids):
            df_cust_rfm = df_RFM[df_RFM['Member_number'].isin(selected_ids)].sort_values(['Member_number'], ascending= False, ignore_index= True)
            st.write(f"#### Khách hàng đã chọn thuộc nhóm")
            st.dataframe(df_cust_rfm[['Member_number', 'CustGroup', 'Recency', 'Frequency', 'Monetary']])
            filtered_df_new = df[df['Member_number'].isin(selected_ids)].sort_values(['Member_number', 'Date'], ascending= False, ignore_index= True)
            
            st.write("#### Khoảng chi tiêu ($):")
            grosssale = filtered_df_new.groupby('Member_number').agg({'Sales': ['min', 'max', 'sum']}).reset_index()
            grosssale.columns = ['Member_number', 'Min', 'Max', 'Total']
            st.dataframe(grosssale)
            
            st.write("#### Thông tin mua hàng sắp xếp theo lần gần nhất:")
            st.dataframe(filtered_df_new)

            def cust_top_product(selected_ids):
                df_choose = df[df['Member_number'].isin(selected_ids)]
                df_top = df_choose.groupby(['productName','price']).value_counts().sort_values(ascending=False).head(20).reset_index()
                return df_top
            def text_underthesea(text):
                products_wt = text.str.lower().apply(lambda x: word_tokenize(x, format="text"))
                products_name_pre = [[text for text in set(x.split())] for x in products_wt]
                products_name_pre = [[re.sub('[0-9]+','', e) for e in text] for text in products_name_pre]
                products_name_pre = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml']]
                                    for text in products_name_pre] # ký tự đặc biệt
                products_name_pre = [[t for t in text if not t in stop_words] for text in products_name_pre] # stopword
                return products_name_pre
            def wcloud_visualize(input_text):
                flat_text = [word for sublist in input_text for word in sublist]
                text = ' '.join(flat_text)
                wc = WordCloud(
                                background_color='white',
                                colormap="ocean_r",
                                max_words=50,
                                width=1600,
                                height=900,
                                max_font_size=400)
                wc.generate(text)
                plt.figure(figsize=(8,12))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            top_products = cust_top_product(selected_ids)
            st.write("#### Top 5 đơn hàng của được mua nhiều nhất của KH được chọn:")
            st.dataframe(top_products.head())
            st.write("#### Word Cloud sản phẩm được mua nhiều nhất của KH được chọn:")
            sample_text = top_products['productName']
            processed_text = text_underthesea(sample_text)
            wcloud_visualize(processed_text)

        else:
            # Không có khách hàng
            st.write("Vui lòng chọn ID ở khung trên :rocket:")


    elif type == "Dự đoán KH mới":
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("Model thực hiện dự đoán phân cụm cho KH mới thông qua hành vi mua hàng (bởi các thông số R, F, M)")
        st.write("##### 2. Thông tin khách hàng")
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng mới (Tối đa 3 KH)")
        # Loop to get input from the user for each customer
            # Get input using sliders
        # Tạo DataFrame rỗng
        df_customer = pd.DataFrame(columns=["Member_number_Predict", "Recency", "Frequency", "Monetary"])

        # Lặp qua 3 khách hàng
        for i in range(3):
            st.write(f"##### Khách hàng {i+1}")
            Member_number_Predict = i+1
            # Sử dụng sliders để nhập giá trị cho Recency, Frequency và Monetary
            recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary", 1, 1000, 100, key=f"monetary_{i}")
            
            # Thêm dữ liệu nhập vào DataFrame
            df_customer = df_customer.append({"Member_number_Predict": Member_number_Predict, "Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)
            
        # Hiển thị DataFrame
        st.dataframe(df_customer)
        df_new_predict = cus_segment_model.predict(df_customer[['Recency','Frequency','Monetary']])
        df_customer["Cluster"] = df_new_predict
        conditions3 = [
        (df_customer['Cluster'] == 0),
        (df_customer['Cluster'] == 1),
        (df_customer['Cluster'] == 2),
        (df_customer['Cluster'] == 3),
        (df_customer['Cluster'] == 4)
        ]
        choices3 = ['REGULARS', 'NEW', 'VIP', 'LOYAL', 'LOST']
        df_customer['CustGroup_Predict'] = np.select(conditions3, choices3, default='UNKNOWN')
        st.dataframe(df_customer)

with tab5:
    st.image('Data/marketing-strategy.jpg')
    st.write('#### KH MỚI (NEW):')
    st.write('KH mới giao dịch gần đây số tiền và tần suất không cao -> có thể tiềm năng nếu đẩy thêm nhiều chính sách dành cho KH mới.')
    st.write('#### KH VIP (VIP):')
    st.write('KH giao dịch nhiều+gần với ngày báo cáo, với tần suất lớn và chi số tiền lớn -> nhóm rất tiềm năng, cần đưa ra các chính sách ưu đãi để khuyến khích KH mua nhiều hơn')
    st.write('#### KH THÂN THIẾT (LOYAL):')
    st.write('KH giao dịch không lâu trước đây, nhưng tần suất và số tiền ở mức trung bình -> đã bắt đầu có thói quen mua hàng tại đơn vị này, tiềm năng có thể nâng hạng lên KH VIP, nên theo dõi và chăm sóc khách hàng nhiều hơn.')
    st.write('#### KH THÔNG THƯỜNG (REGULARS):')
    st.write('KH giao dịch khá lâu trước đây, nhưng tần suất và số tiền ở mức trung bình  -> có thể là nhóm KH khi có nhu cầu thì tiện thể ghé mua, không tiềm năng.')
    st.write('#### KH RỜI BỎ (LOST):')
    st.write('KH đã rất lâu không giao dịch và tần suất + số tiền thấp -> không tiềm năng')
    
