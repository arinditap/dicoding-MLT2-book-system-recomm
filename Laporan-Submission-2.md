# Laporan Proyek Machine Learning - Arindita Prihastama

## _"Sistem Rekomendasi Buku"_

## Project Overview

Membaca buku merupakan salah satu cara bagi setiap orang untuk mendapatkan ilmu pengetahuan dan wawasan baru mengenai banyak hal. Buku merupakan sumber informasi dan ilmu pengetahuan yang dapat meningkatkan wawasan tentang berbagai macam hal. "Buku adalah jendela dunia". Pepatah tersebut menunjukkan bahwasannya membaca itu penting. Informasi mengenai beragam jenis buku saat ini telah tersebar luas di internet sehingga dapat diakses oleh banyak orang. Meskipun informasi mengenai beragam jenis buku telah beredar di internet, namun minat baca di Indonesia masih tergolong rendah. Salah satu hal yang menyebabkan hal tersebut adalah banyaknya buku yang ada sehingga pembaca kesulitan untuk menemukan buku yang sesuai dengan kriteria yang diinginkan pembaca. 

Sebagai salah satu upaya untuk meningkatkan minat baca masyarakat adalah dengan membangun sistem rekomendasi buku untuk memudahkan pembaca mencari buku yang sesuai kriteria yang diinginkan dan buku-buku serupa. Untuk itu, dapat diterapkan metode _Content-Based Filtering_ dan _Collaborative Filtering_ ke dalam sistem rekomendasi buku ini. Apabila pembaca mendapatkan rekomendasi buku yang sesuai, diharapkan pembaca dapat mengakses informasi yang sesuai dengan yang diinginkan atau buku-buku dengan kriteria serupa, sehingga minat baca di masyarakat juga mengalami kenaikan.

Referensi:
  - [Sistem Rekomendasi Buku pada Perpustakaan Daerah Provinsi Kalimantan Selatan Menggunakan Metode Content-Based Filtering](https://journal.universitasbumigora.ac.id/index.php/matrik/article/view/617)
  - [SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612).
  - [Sistem Rekomendasi Buku Menggunakan Metode Content Based Filtering](https://dspace.uii.ac.id/handle/123456789/35942).
  - [Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-Based Filtering](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/18066/17695)

## Business Understanding

Penelitian ini dilakukan dengan tujuan memberikan rekomendasi buku sesuai dengan kriteria dari pembaca melalui sebuah sistem rekomendasi buku. Sistem rekomendasi dibuat dengan menerapkan content-based filtering dan collaborative filtering.

### Problem Statements
- Bagaimana membuat sistem rekomendasi buku berdasarkan kriteria dari pembaca menggunakan content-based filtering?
- Bagaimana membuat sistem rekomendasi buku yang mungkin akan disukai berdasarkan rating menggunakan collaborative filtering?

### Goals

- Memberikan sejumlah rekomendasi buku yang sesuai dengan kriteria pembaca atau buku yang serupa dengan menggunakan content-based filtering.
- Memberikan sejumlah rekomendasi buku yang belum dibaca atau mungkin akan disukai oleh pembaca dengan menggunakan collaborative filtering.

    ### Solution statements
    - Menerapkan content-based filtering dengan menggunakan tf-idf dan cosine similarity untuk mendapatkan top 5 dari kemiripan buku berdasarkan judul buku dan penulis buku.
    - Menerapkan collaborative filtering dengan menggunakan class RecommenderNet untuk mendapatkan rekomendasi top 10 buku dengan rating tinggi.

## Data Understanding
Dataset yang digunakan diambil dari situs **Kaggle** yang berjudul [_"goodbooks-10k"_](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=books.csv). Dataset ini berisi 5 file dengan ekstensi csv, yaitu:
1. `book_tags.csv`, file ini berisi data tag buku (label), diurutkan berdasarkan ascending goodreadsbookid dan count descending.
   - goodreads_id : ID dari goodreads
   - tag_id : ID tag (genre)
   - count : Jumlah goodreads

  Untuk mengetahui jumlah genre:  
  ```sh
    print('Jumlah genre buku: ', len(genre.tag_id.unique()))
  ```
  Output: 
  ```sh
    Jumlah genre buku:  34252
  ```
2. `books.csv`, berisi informasi mengenai buku.
   - id : ID dari file books
   - book_id : ID buku
   - best_book_id : ID dari buku populer
   - work_id : ID karya
   - books_count : jumlah edisi buku tertentu
   - isbn : nomor isbn
   - authors : nama penulis
   - original_publication_year : tahun terbit buku
   - original_title : judul asli buku
   
   Untuk melihat jumlah buku dan jumlah nama penulis menggunakan:
    ``` sh
        print('Jumlah buku: ', len(buku.book_id.unique()))
        print('Jumlah author: ', len(buku.authors.unique()))
    ```
    Output:
   ```sh
     Jumlah buku:  10000
     Jumlah author:  4664
   ```
4. `ratings.csv`, berisi rating buku sesuai id pengguna.
   - book_id : ID buku
   - user_id : ID Pengguna
   - rating : rating buku
   
   Jumlah tiap rating dapat dilihat pada Gambar 1
   ![download](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/1605dd73-96c3-43e3-b268-71557daa7d91)
   > Gambar 1. Visualisasi Jumlah Rating Buku dari Pengguna
5. `tag.csv`, berisi tentang pemetaan id-nama tag.
   - tag_id : ID tag (genre)
   - tag_name : Nama tag (genre)

   Untuk mengetahui jumlah genre:  
    ```sh
      print('Jumlah jenis genre buku: ', len(jenis_genre.tag_id.unique()))
    ```
   Output: 
    ```sh
      Jumlah jenis genre buku:  34252
    ```
6. `to_read.csv`, daftar buku yang ditandai oleh pengguna "untuk dibaca". Diurutkan berdasarkan user_id dan book_id.
   - user_id : ID pengguna/pembaca
   - book_id : ID buku
   
    Untuk mengetahui jumlah genre:  
    ```sh
      print('Jumlah user: ', len(user.user_id.unique()))
    ```
    Output: 
    ```sh
      Jumlah user:  48871
    ``` 
  
File yang digunakan untuk pemodelan adalah file `books.csv` dan `ratings.csv` dengan menggunakan variabel **'book_id', 'original_title', 'authors', 'original_publication_year'** untuk model _content-based filtering_, dan variabel **'user_id', 'book_id', 'authors', 'original_title'** untuk model _collaborative filtering_.


## Data Preparation
Teknik yang digunakan untuk data preparation antara lain sebagai berikut:
1. **Mengatasi Missing Value**, dilakukan pengecekan data apakah ada data yang kosong dan berapa jumlahnya menggunakan `isnull().sum()`, sehingga didapatkan hasil:
   ```sh
     book_id                             0
     user_id                             0
     rating                              0
     authors                      88860317
     title                        88860317
     original_publication_year    88870317
     dtype: int64
   ```
    Dari output didapatkan bahwa terdapat missing value pada variabel _authors_, _title_, dan _original_publication_year_. Kemudian dilakukan penghapusan data dengan missing value menggunakan `dropna()` untuk menghilangkan data yang bernilai kosong. Selanjutnya merapikan data dengan menghapus data yang sama dengan `drop_duplicates()`.
   
   Teknik mengatasi missing value ini diterapkan untuk data preparation pada model _content based filtering_.
2. **Encoding Data**, melakukan penyandian (encoding) fitur **'user_id'** dan **book_id** ke bentuk indeks integer. Cek data setelah encoding fitur menghasilkan:
   ```sh
      53424
      10000
      Jumlah Pembaca: 53424, Jumlah Buku: 10000, Min Rating: 1.0, Max Rating: 5.0
   ```
3. **Membagi Dataset**, dataset akan dibagi menjadi data train dan data test dengan rasio 80:20. Namun sebelum itu dilakukan pengacakan dataset agar distribusinya random/acak. Teknik Encoding Data dan Membagi Dataset dilakukan untuk nantinya dipakai pada model _collaborative filtering_.

## Modeling
1. _**Content Based Filtering**_. Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi. Pada model ini diterapkan teknik _TF-IDF_ untuk mengubah fitur text menjadi fitur numerik dengan menggunakan fungsi `TfidfVectorizer()` dari library sklearn.

   Kelebihan: baik dipakai ketika skala user besar, dapat menemukan ketertarikan spesifik dari seorang user dan dapat merekomendasikan item yang jarang disukai orang lain.

   Kekurangan: karena meta feature yang digunakan kita yang menentukan sendiri, kualitas dari rekomendasi tergantung kualitas dari meta feature itu sendiri.

   Modelling menggunakan content based filtering menghasilkan rekomendasi top 5 buku dari judul buku yang dimasukkan pada sistem.
   ![image](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/95aded26-02ed-45e2-a205-990b3a2a7f37)
   > Gambar 2. Hasil Rekomendasi Top 5 Buku dari Judul Buku yang dimasukkan.

   Dari Gambar 2 dapat dilihat saat kita memasukkan judul buku `The Door Into Summer` muncul informasi buku tersebut dengan nama penulis `Robert A. Heinlein`. Kemudian setelah dilakukan pemanggilan fungsi `book_recommendation()` dengan judul tersebut muncul rekomendasi buku dengan penulis sama, karena sistem rekomendasi dibuat berdasarkan penulis.

2. _**Collaborative Filtering**_. Collaborative filtering bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Model ini memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya, misalnya cara pemberian rating terhadap suatu item. Metode ini merekomendasikan item-item yang dipilih oleh pengguna lain dengan kemiripan model item dari pengguna saat ini.

   Kelebihan: Hasil rekomendasi yang beragam dan bersifat _serendipitous_ (relevan dan baru)

   Kekurangan:
   - Cold-start problem (tidak dapat menghasilkan rekomendasi dikarenakan tidak adanya informasi preferensi) untuk pengguna baru dan item baru
   - Sparse problem (matriks rating pengguna-item yang jarang/banyak yang kosong dapat mempengaruhi keakuratan algoritma)

   Modelling menggunakan collaborative filtering menghasilkan rekomendasi top 10 buku dengan rating tinggi, dapat dilihat pada Gambar 3.
   ![image](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/0d96ff8a-dba7-42d5-ab15-2c8e2e7cabe3)
   > Gambar 3. Hasil Rekomendasi Top 10 Buku dengan Rating Tinggi.

## Evaluation
_**Root Mean Squared Error**_

Metrik RMSE _(Root Mean Square Error)_ digunakan pada penelitian ini untuk mengevaluasi kinerja model yang dihasilkan. RMSE merupakan cara standar untuk mengukur kesalahan model dalam memprediksi data kuantitatif. _Root Mean Squared Error_ (RMSE) mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Perhitungan RMSE ditunjukkan pada rumus dalam Gambar 4.

![image](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/b7271fa3-2ceb-4935-873d-d77516843d44)
> Gambar 4. Rumus Root Mean Square Error

Keterangan: 

RMSE = nilai root mean square error

y = nilai hasil observasi

ŷ = nilai hasil prediksi

i = urutan data

n = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa bedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. Metrik diterapkan dengan kode berikut:
```sh
  model = RecommenderNet(num_readers, num_books, 50) # inisialisasi model

  # model compile
  model.compile(
      loss = tf.keras.losses.BinaryCrossentropy(),
      optimizer = keras.optimizers.Adam(learning_rate=0.001),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
  )
```
Kemudian proses training dilakukan sebagai berikut: 
```sh
  history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    steps_per_epoch = 10000,
    epochs = 15,
    validation_data = (x_val, y_val)
  )
```
Output dari proses training dapat dilihat pada Gambar 5.

![image](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/abee4f8f-9764-4525-904c-9075f2f3c732)
> Gambar 5. Output Proses Training

Visulaisasi evaluasi metrik menggunakan RMSE setelah pelatihan untuk model Collaborative filtering dapat dilihat pada Gambar 6.

![rmse](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/dd8c29bd-b18f-4066-84a5-b7ce256cae4b)
> Gambar 6. Visualisasi Hasil RMSE