# Model dogrulama

Uretim: `python scripts/validate_model.py` (varsayilan `BaselineGains`, `ShieldParams`, `GridParams`, `BessParams`).
Asagidaki cikti betigin birebir ciktisidir. `config.py` degerlerinde degisiklik yapilmadi.

## 1. Yuk adimi tablosu (t=10 s, 150 s, gurultusuz)

### Adim = 0.02 pu (yuk artisi)

| Kontrolcu | nadir [mHz] | yerlesme [s] | throughput [kWh] | max abs p_ess [pu] | son abs df [mHz] | kalkan aktif [%] |
|---|---|---|---|---|---|---|
| NoBESS | -93.3 | 8.4 | 0.000 | 0.000 | 0.0 | 0.0 |
| Droop | -36.3 | 0.8 | 0.391 | 0.029 | 0.0 | 0.0 |
| PI | -36.3 | 0.3 | 0.555 | 0.030 | 0.0 | 0.0 |
| PID | -25.1 | 0.2 | 0.555 | 0.031 | 0.0 | 0.0 |
| ZeroAgent+Shield | -93.3 | 8.4 | 0.000 | 0.000 | 0.0 | 0.0 |

### Adim = 0.05 pu (yuk artisi)

| Kontrolcu | nadir [mHz] | yerlesme [s] | throughput [kWh] | max abs p_ess [pu] | son abs df [mHz] | kalkan aktif [%] |
|---|---|---|---|---|---|---|
| NoBESS | -233.1 | 17.3 | 0.000 | 0.000 | 0.0 | 0.0 |
| Droop | -90.7 | 5.6 | 0.978 | 0.072 | 0.0 | 0.0 |
| PI | -90.7 | 1.6 | 1.388 | 0.074 | 0.0 | 0.0 |
| PID | -62.8 | 1.4 | 1.388 | 0.078 | 0.0 | 0.0 |
| ZeroAgent+Shield | -206.4 | 17.3 | 0.049 | 0.044 | 0.0 | 0.7 |

### Adim = 0.10 pu (yuk artisi)

| Kontrolcu | nadir [mHz] | yerlesme [s] | throughput [kWh] | max abs p_ess [pu] | son abs df [mHz] | kalkan aktif [%] |
|---|---|---|---|---|---|---|
| NoBESS | -466.3 | 24.1 | 0.000 | 0.000 | 0.0 | 0.0 |
| Droop | -181.5 | 12.2 | 1.955 | 0.143 | 0.0 | 0.0 |
| PI | -181.5 | 2.8 | 2.777 | 0.148 | 0.0 | 0.0 |
| PID | -146.4 | 2.8 | 2.762 | 0.154 | 0.0 | 0.0 |
| ZeroAgent+Shield | -359.3 | 24.1 | 0.336 | 0.116 | 0.0 | 3.4 |

### Adim = 0.15 pu (yuk artisi)

| Kontrolcu | nadir [mHz] | yerlesme [s] | throughput [kWh] | max abs p_ess [pu] | son abs df [mHz] | kalkan aktif [%] |
|---|---|---|---|---|---|---|
| NoBESS | -699.4 | 28.1 | 0.000 | 0.000 | 0.0 | 0.0 |
| Droop | -272.2 | 16.1 | 2.924 | 0.189 | 0.0 | 0.0 |
| PI | -272.2 | 3.6 | 4.105 | 0.189 | 0.0 | 0.0 |
| PID | -265.1 | 3.5 | 4.112 | 0.184 | 0.0 | 0.0 |
| ZeroAgent+Shield | -420.9 | 28.0 | 1.005 | 0.184 | 0.0 | 6.1 |

## 2. Kararlilik taramasi (0.10 pu adim)
Olcut: son 20 s icinde max abs df < 50 mHz ve sonlu.

| kp | Droop son-pencere max [mHz] | Droop kararli | PI son-pencere max [mHz] | PI kararli |
|---|---|---|---|---|
| 1 | 0.0 | True | 0.0 | True |
| 2 | 0.0 | True | 0.0 | True |
| 3 | 0.0 | True | 0.0 | True |
| 4 | 0.0 | True | 0.0 | True |
| 5 | 0.0 | True | 0.0 | True |
| 6 | 0.0 | True | 0.0 | True |
| 7 | 0.0 | True | 0.0 | True |
| 8 | 0.0 | True | 0.0 | True |
| 9 | 1.3 | True | 184.8 | False |
| 10 | 184.8 | False | 184.8 | False |
| 11 | 184.8 | False | 184.8 | False |
| 12 | 184.8 | False | 184.8 | False |

Kararli Droop kp araligi: 1 .. 9

Kararli PI kp araligi: 1 .. 8

## 3. 20 rastgele egitim senaryosu (PI varsayilan, sifir-ajan+kalkan)

PI max abs df [mHz]: min 34.5, q25 97.8, medyan 150.1, q75 167.3, q90 174.2, max 182.7
PI ile max abs df > engage (150 mHz) olan senaryo: 10/20
Sifir-ajan kalkani tetikledi: 18/20; ortalama aktif sure %3.3

## Tartisma

**Fizik tutarli.** NoBESS nadiri yuk adimiyla dogrusal olceklenir (0.02 pu icin -93 mHz, 0.10 pu icin -466 mHz;
bu, yaklasik 1/beta = 1/21 pu/pu * 50 Hz = 2.4 Hz/pu duragan durum egimine uygundur). AGC sayesinde butun
kontrolcuelerde son sapma 0'dir ve BESS gucu sifira doner. BESS nadiri belirgin dusurur (0.10 pu: -466 -> -146...-181 mHz).

**Droop ve PI ayni nadiri verir.** Nadir ilk saniyelerde olusur; integral terimi (ki=2) bu sure icinde henuz
birikmez. PI'nin farki yerlesme suresindedir (0.10 pu: 12.2 s -> 2.8 s). PID turevi nadiri ancak
belirgin azaltir (-146 mHz) fakat 0.15 pu adimda guc sinirina (p_max=0.2) takildigi icin kazanc kucuktur.
Buyuk adimlarda (0.15 pu) p_max doyumu nadiri belirler; kontrolcu ayari artik etkisizdir.

**Kalkan.** Sifir-ajan, 0.02 pu adimda kalkani hic tetiklemez (nadir 93 mHz < 150 mHz). 0.05 pu ve ustunde tetikler.
Kalkan devredeyken frekansi yalnizca kismen toparlar (nadir -359 mHz @ 0.10 pu), cunku engage esigi asildiktan
sonra devreye girer; bu bir "son care" sigortasidir, performans kontrolcusu degil.

**Kararli kazanc araliklari (0.10 pu adim, son 20 s'de |df| < 50 mHz).**
Droop kp 1..9, PI kp 1..8 (ki=2). kp >= 9-10 civarinda limit cevrim baslar (son penceredeki genlik 185 mHz).
**Duzeltme (hakem incelemesi):** 2 pu/s'de `ramp_rate*dt = 0,2 pu = p_max` oldugundan hiz siniri pratikte baglayici
degildir; bu kalan limit cevrimin nedeni p_max doyumu ile ornekleme gecikmesidir (dt = T_ess = 0,1 s). Hiz siniri
0,4 pu/s iken ise orta kazancli PI'da da limit cevrim olusuyordu (orkestrator prototipi); 2 pu/s'ye cikarmanin nedeni
budur. Fiziksel gerekce: sebeke olcekli BESS eviricileri tam guce 100–200 ms'de cikabilir (hizli frekans tepkisi
urunleri <= 1 s ister), bu nedenle 2 MW'a 0,1 s'de ulasmak gercekcidir.
Tarama kaba (tam sayi adimlar) oldugundan sinir +-1 belirsizdir.
`tune_baselines.py` onerisi: `droop_k` in [0.5, 9], `pi_kp` in [0.5, 8], `pid_kp` in [0.5, 8]; `pi_ki`/`pid_ki`
icin bu taramada ki=2 sabit tutuldu, aralik [0.1, 5] makul bir baslangictir ve dogrulanmamistir;
`pid_kd` icin [0, 2] dogrulanmamis baslangic araligidir. Kararsiz kazanclar zaten buyuk maliyet uretir.

**Rastgele senaryolar (20 egitim senaryosu).** PI'nin max |df| dagilimi: medyan ~150 mHz, q90 ~174 mHz,
max 183 mHz; 20 senaryonun 10'unda PI bile 150 mHz'i asar. Sifir-ajan kalkani 18/20 senaryoda tetikler, ortalama
aktif sure %3.3.

**150 mHz esigi hakkinda.** Egitim dagiliminda adim buyuklugu 0.02-0.10 pu ve ust uste binen ruzgar/gurultu
vardir; iyi ayarli bir PI bile senaryolarin yarisinda 150 mHz'i asar. Dolayisiyla esik "yalnizca ajanin hatasi"
degil, "buyuk bozucu" durumlarini da yakalar: kalkan yakin ornekler icin kalici olarak ajanin yerine gecer ve
`autonomy_pct` zorlu senaryolarda dusuk cikar. Esigin ajani cezalandirmasi adil degildir; seceneker:
(a) engage_hz'i 200-250 mHz'e cikarmak, (b) mevcut esigi tutup `autonomy_pct`'yi senaryo zorluguyla birlikte raporlamak.
Bu asamada `config.py` degistirilmedi, karar orkestratore birakilir.

## 4. Orkestrator karari: kalkan esigi 150 -> 200 mHz

Karar **egitim bolmesine** dayanir (§3 ve asagidaki egitim satirlari): varsayilan PI'in en kotu sapmasi 182 mHz, PID'inki
155 mHz; 200 mHz ikisinin de ustundedir. Test satirlari yalnizca **dogrulama** amaciyla eklenmistir ve karari degistirmez
(hakem incelemesi, test bankasinin karar metninde kullanilmasini usul hatasi olarak not etti). ENTSO-E 200 mHz degeri
kita senkron alani icindir; 10 MW'lik ada sebekesi icin yalnizca bir referans buyukluk olarak alinmistir.

Varsayilan kazanclarla olculen en kotu sapma:

| Bolme | Kontrolcu | medyan max abs df | q90 | en buyuk | >150 mHz | >200 mHz |
|---|---|---|---|---|---|---|
| egitim | PI  | 150 | 173 | 182 | %50 | %0 |
| egitim | PID | 102 | 133 | 155 | %2  | %0 |
| test   | PI  | 135 | 173 | 185 | %32 | %0 |
| test   | PID |  96 | 138 | 150 | %0  | %0 |

Sifir-ajan + kalkan, test bankasinda: 150 mHz esikte 38/40 senaryoda, 200 mHz esikte 35/40 senaryoda tetikleniyor.
0.10 pu adimda kalkanli sifir-ajanin tepe sapmasi her iki esikte de 466 -> 359 mHz.

Karar: `ShieldParams.engage_hz = 0.20` (release 0.10 ve min_hold 1 s ayni). 200 mHz ENTSO-E'nin azami kalici
frekans sapmasidir ve iki bolmede de varsayilan PI/PID'in en kotu degerinin ustundedir. Boylece kalkan, ajan makul bir
klasik kontrolcuden kotu davrandiginda devreye girer; `autonomy_pct` bozucunun buyuklugunu degil ajanin guvenilirligini olcer.
