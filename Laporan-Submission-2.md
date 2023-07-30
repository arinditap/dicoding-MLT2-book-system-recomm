# Laporan Proyek Machine Learning - Arindita Prihastama

## _"Sistem Rekomendasi Buku"_

## Project Overview

Membaca buku merupakan salah satu cara bagi setiap orang untuk mendapatkan ilmu pengetahuan dan wawasan baru mengenai banyak hal. Buku merupakan sumber informasi dan ilmu pengetahuan yang dapat meningkatkan wawasan tentang berbagai macam hal. "Buku adalah jendela dunia". Pepatah tersebut menunjukkan bahwasannya membaca itu penting. Informasi mengenai beragam jenis buku saat ini telah tersebar luas di internet sehingga dapat diakses oleh banyak orang. Meskipun informasi mengenai beragam jenis buku telah beredar di internet, namun minat baca di Indonesia masih tergolong rendah[3]. Salah satu hal yang menyebabkan hal tersebut adalah banyaknya buku yang ada sehingga pembaca kesulitan untuk menemukan buku yang sesuai dengan kriteria yang diinginkan pembaca. 

Sebagai salah satu upaya untuk meningkatkan minat baca masyarakat adalah dengan membangun sistem rekomendasi buku untuk memudahkan pembaca mencari buku yang sesuai kriteria yang diinginkan dan buku-buku serupa. Untuk itu, dapat diterapkan metode _Content-Based Filtering_ dan _Collaborative Filtering_ ke dalam sistem rekomendasi buku ini. Apabila pembaca mendapatkan rekomendasi buku yang sesuai, diharapkan pembaca dapat mengakses informasi yang sesuai dengan yang diinginkan atau buku-buku dengan kriteria serupa, sehingga minat baca di masyarakat juga mengalami kenaikan.


## Business Understanding

Penelitian ini dilakukan dengan tujuan memberikan rekomendasi buku sesuai dengan kriteria dari pembaca melalui sebuah sistem rekomendasi buku. Sistem rekomendasi dibuat dengan menerapkan content-based filtering dan collaborative filtering.

### Problem Statements
- Bagaimana membuat sistem rekomendasi buku berdasarkan kriteria dari pembaca menggunakan content-based filtering?
- Bagaimana membuat sistem rekomendasi buku yang mungkin akan disukai berdasarkan rating menggunakan collaborative filtering?

### Goals

- Memberikan sejumlah rekomendasi buku yang sesuai dengan kriteria pembaca atau buku yang serupa dengan menggunakan content-based filtering.
- Memberikan sejumlah rekomendasi buku yang belum dibaca atau mungkin akan disukai oleh pembaca dengan menggunakan collaborative filtering.
- Evaluasi terhadap model yang digunakan.

    ### Solution statements
    - Menerapkan content-based filtering dengan menggunakan tf-idf dan cosine similarity untuk mendapatkan top 5 dari kemiripan buku berdasarkan judul buku dan penulis buku.
    - Menerapkan collaborative filtering dengan menggunakan class RecommenderNet untuk mendapatkan rekomendasi top 10 buku dengan rating tinggi.
    - Melakukan evaluasi model dengan menggunakan precision dan RMSE.

## Data Understanding
Dataset yang digunakan diambil dari situs **Kaggle** yang berjudul [_"goodbooks-10k"_](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=books.csv). Dataset ini berisi 5 file dengan ekstensi csv, yaitu:
1. **'book_tags.csv'**, file ini berisi data tag buku (label), diurutkan berdasarkan ascending goodreadsbookid dan count descending.
   - goodreads_id : ID dari goodreads
   - tag_id : ID tag (genre)
   - count : Jumlah goodreads

  Untuk mengetahui jumlah genre digunakan fungsi `len()` dengan menambah fungsi `unique`. Sehingga outputnya: 
  ```sh
    Jumlah genre buku:  34252
  ```
2. **'books.csv'**, berisi informasi mengenai buku.
   - id : ID dari file books
   - book_id : ID buku
   - best_book_id : ID dari buku populer
   - work_id : ID karya
   - books_count : jumlah edisi buku tertentu
   - isbn : nomor isbn
   - authors : nama penulis
   - original_publication_year : tahun terbit buku
   - original_title : judul asli buku
   
   Untuk melihat jumlah buku dan jumlah nama penulis menggunakan fungsi `len()` dan`unique()`. Sehingga outputnya:
   ```sh
     Jumlah buku:  10000
     Jumlah author:  4664
   ```
4. **'ratings.csv'**, berisi rating buku sesuai id pengguna.
   - book_id : ID buku
   - user_id : ID Pengguna
   - rating : rating buku
   
   Jumlah tiap rating dapat dilihat pada Gambar 1
   ![download](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/1605dd73-96c3-43e3-b268-71557daa7d91)
   > Gambar 1. Visualisasi Jumlah Rating Buku dari Pengguna
5. **'tag.csv'**, berisi tentang pemetaan id-nama tag.
   - tag_id : ID tag (genre)
   - tag_name : Nama tag (genre)

   Untuk mengetahui jumlah genre menggunakan fungsi `len()` dan `unique()`. Outputnya:
    ```sh
      Jumlah jenis genre buku:  34252
    ```
6. **'to_read.csv'**, daftar buku yang ditandai oleh pengguna "untuk dibaca". Diurutkan berdasarkan user_id dan book_id.
   - user_id : ID pengguna/pembaca
   - book_id : ID buku
   
    Dari data didapatkan Jumlah User = 48871.
      
File yang digunakan untuk pemodelan adalah file 'books.csv' dan 'ratings.csv' dengan menggunakan variabel **'book_id', 'original_title', 'authors', 'original_publication_year'** untuk model _content-based filtering_, dan variabel **'user_id', 'book_id', 'authors', 'original_title'** untuk model _collaborative filtering_.


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
1. _**Content Based Filtering**_. Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi. Pada model ini diterapkan teknik _TF-IDF_ untuk mengubah fitur text menjadi fitur numerik dengan menggunakan fungsi `TfidfVectorizer()` dari library sklearn[1].

   Kelebihan: baik dipakai ketika skala user besar, dapat menemukan ketertarikan spesifik dari seorang user dan dapat merekomendasikan item yang jarang disukai orang lain.

   Kekurangan: karena meta feature yang digunakan ditentukan sendiri, kualitas dari rekomendasi tergantung kualitas dari meta feature itu sendiri.

   Modelling menggunakan content based filtering menghasilkan rekomendasi top 5 buku dari judul buku yang dimasukkan pada sistem.

   > Tabel 1. Hasil Pencarian Informasi Berdasarkan Judul Buku

   |    | IDBuku |           judul_buku |            penulis | thn_terbit |
   |----|--------|----------------------|--------------------|------------|
   | 39 |   348  | The Door Into Summer | Robert A. Heinlein |     1957.0 |

   > Tabel 2. Hasil Rekomendasi Top 5 Buku dari Judul Buku yang Dimasukkan
   
   |   |                                        judul_buku |            penulis |
   |---|---------------------------------------------------|--------------------|
   | 0 |                              Time Enough for Love | Robert A. Heinlein |
   | 1 |                        Stranger in a Strange Land | Robert A. Heinlein |
   | 2 |                          Job: A Comedy of Justice | Robert A. Heinlein |
   | 3 | Mrs. Frisby and the Rats of NIMH (Rats of NMH ... |  Robert C. O'Brien |
   | 4 |                                    Shadow Drivers |      Robert Kurson |

   Dari Tabel 1 dapat dilihat saat memasukkan judul buku **'The Door Into Summer'** muncul informasi buku tersebut dengan nama penulis **'Robert A. Heinlein'**. Kemudian setelah dilakukan pemanggilan fungsi `book_recommendation()` dengan judul tersebut muncul rekomendasi buku dengan penulis sama, karena sistem rekomendasi dibuat berdasarkan penulis seperti yang tertera pada Tabel 2.

2. _**Collaborative Filtering**_. Collaborative filtering bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Model ini memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya, misalnya cara pemberian rating terhadap suatu item. Metode ini merekomendasikan item-item yang dipilih oleh pengguna lain dengan kemiripan model item dari pengguna saat ini[2].

    Pada tahap ini, model akan menghitung skor kecocokan antara pembaca(user) dan buku menggunakan tekenik embedding. Langkah yang pertama adalah melakukan proses embedding terhadap data pembaca(user) dan buku. Kemudian lakukan perkalian dot antara embedding pembaca dan buku. Tambahkan bias untuk setiap pembaca dan buku. Fungsi sigmoid digunakan untuk menetapkan skor kecocokan dalam skala [0,1]. Langkah-langkah yang disebutkan diatas akan dimasukkan ke dalam sebuah class dari Keras bernama 'RecommenderNet()'.

   Kelebihan: Hasil rekomendasi yang beragam dan bersifat _serendipitous_ (relevan dan baru)[4]

   Kekurangan:
   - Cold-start problem (tidak dapat menghasilkan rekomendasi dikarenakan tidak adanya informasi preferensi) untuk pengguna baru dan item baru
   - Sparse problem (matriks rating pengguna-item yang jarang/banyak yang kosong dapat mempengaruhi keakuratan algoritma)[4].

   Modelling menggunakan collaborative filtering menghasilkan rekomendasi top 10 buku dengan rating tinggi, dapat dilihat pada Tabel 3.

   > Tabel 3. Hasil Rekomendasi Top 10 Buku dengan Rating Tinggi

   
   | penulis                              | judul_buku                                                        |
   |--------------------------------------|-------------------------------------------------------------------|
   | Don DeLillo                          | Libra                                                             |
   | Paul Auster                          | The Brooklyn Follies                                              |
   | Robert M. Pirsig                     | Zen and the Art of Motorcycle Maintenance: An Inquiry Into Values |
   | h.G. Wells, Greg Bear, Carlo Pagetti | The Time Machine                                                  |
   | Tom Wolfe                            | The Bonfire of the Vanities                                       |
   | Jacquelyn Mitchard                   | The Deep End of The Oceaan (Cappadora Family, #1)                 |
   | Toni Morison                         | Paradise                                                          |
   | V.S. Naipaul                         | A Bend in the River                                               |
   | Pat Barker                           | Regeneration (Regeneration, #1)                                   |
   | Fyodor Dostoyevsky, David McDuff     | Crime and Punishment                                              |
   
## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah _Precision_ dan _Root Mean Square Error_. Untuk model content based filtering akan dievaluasi menggunakan _Precision_, sedangkan collaborative filtering dievaluasi dengan _RMSE_.

_**Precision**_

Precision merupakan metrik untuk evaluasi model machine learning dimana metrik ini yang mengkuantifikasi jumlah prediksi positif yang benar yang dibuat. Berikut merupakan rumus dari metrik precision:

Precision = num of recommendation that are relevant / num of item we recommended

Diketahui pada output rekomendasi content based filtering didapat rekomendasi buku berdasarkan penulisnya. Penulis yang dicari adalah "Robert A. Heinlein". Sistem memberikan 5 rekomendasi buku, dimana 3 dari 5 buku adalah buku dengan penulis yang sama atau data relevan. Maka precision dicari dengan membagikan hasil rekomendasi yang relevan dibagi jumlah yang direkomendasikan. Jadi precision = 3/5 = 0.6, sehingga persentase presisinya adalah 60%.

_**Root Mean Squared Error**_

Metrik RMSE _(Root Mean Square Error)_ digunakan pada penelitian ini untuk mengevaluasi kinerja model yang dihasilkan. RMSE merupakan cara standar untuk mengukur kesalahan model dalam memprediksi data kuantitatif. _Root Mean Squared Error_ (RMSE) mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Perhitungan RMSE ditunjukkan pada rumus berikut.

$$RMSE = \sqrt{\Sigma_{i=1}^{n}{\frac{(ŷ_i - y_i)^{2}}{n}}}$$

Keterangan: 

RMSE = nilai root mean square error

y = nilai hasil observasi

ŷ = nilai hasil prediksi

i = urutan data

n = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa bedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. 

Visulaisasi evaluasi metrik menggunakan RMSE setelah pelatihan untuk model Collaborative filtering dapat dilihat pada Gambar 2.

![rmse](https://github.com/arinditap/dicoding-MLT2-book-system-recomm/assets/48308725/dd8c29bd-b18f-4066-84a5-b7ce256cae4b)
> Gambar 2. Visualisasi Hasil RMSE

Nilai RMSE semakin menurun seiring bertambahnya epoch, hal ini menunjukkan bahwa rekomendasi semakin mendekati nilai prediksi. Dari visualisasi diatas didapatkan nilai error akhir sebesar 0.2138 dan validasi error sebesar 0.2161 yang bisa dikatakan sudah bagus untuk sebuah sistem rekomendasi. 

## Kesimpulan
- Sistem rekomendasi buku dapat dibuat menggunakan content based filtering dimana rekomendasi berdasarkan penulis, hasil akhir model ini merekomendasikan 5 buku dengan penulis yang sama.
- Sistem rekomendasi buku menggunakan collaborative filtering menghasilkan 10 rekomendasi buku berdasarkan rating tertinggi yang diberikan oleh user.
- Setelah dilakukan evaluasi, model content based filtering dievaluasi menggunakan precision didapatkan precision sistem sebesar 60%.
- Model collaborative filtering dievaluasi menggunakan RMSE didapatkan nilai RMSE sebesar 0.2161.

Referensi:
1. Alkaff, M., Khatimi, H., & Eriadi, A. (2020). Sistem Rekomendasi Buku pada Perpustakaan Daerah Provinsi Kalimantan Selatan Menggunakan Metode Content-Based Filtering. MATRIK: Jurnal Manajemen, Teknik Informatika Dan Rekayasa Komputer, 20(1), 193-202.
2. Irfan, M., & Cahyani, A. D. (2014). Sistem Rekomendasi: Buku Online Dengan Metode Collaborative Filtering. Jurnal Teknologi Technoscientia, 076-84.
3. ZAYYAD, M. R. A. (2021). Sistem Rekomendasi Buku Menggunakan Metode Content Based Filtering.
4. Arfisko, H. H., & Wibowo, A. T. (2022). Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-Based Filtering. eProceedings of Engineering, 9(3).

