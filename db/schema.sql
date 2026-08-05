-- DemoBank'ın (sahte) bankacılık verisi için şema + seed.
--
-- docker-compose'daki postgres servisi bu dosyayı /docker-entrypoint-initdb.d/init.sql
-- olarak bağlıyor; postgres imajı buradaki dosyaları sadece boş bir veri
-- dizininde otomatik çalıştırıyor. scripts/seed_postgres.py aynı dosyayı
-- Docker'sız lokal kullanım için çalıştırıyor, o yüzden her ifade idempotent.
--
-- Aynı iki demo müşteri banking_repository.py'deki SEED_ACCOUNTS ile birebir
-- aynı — hangi backend'e karşı test yazılırsa yazılsın aynı veriyi görüyor.

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,       -- IBAN
    owner_name TEXT NOT NULL,
    balance NUMERIC(14, 2) NOT NULL,
    currency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    card_last4 TEXT PRIMARY KEY,       -- demo basitleştirmesi: tüm fixture'da tekil
    account_id TEXT NOT NULL REFERENCES accounts (account_id),
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts (account_id),
    occurred_at TIMESTAMPTZ NOT NULL,
    description TEXT NOT NULL,
    amount NUMERIC(14, 2) NOT NULL,
    UNIQUE (account_id, occurred_at, description)  -- yeniden seed'lemeyi idempotent yapar
);

CREATE INDEX IF NOT EXISTS idx_transactions_account_occurred
    ON transactions (account_id, occurred_at DESC);

INSERT INTO accounts (account_id, owner_name, balance, currency) VALUES
    ('TR330006100519786457841326', 'Ayşe Yılmaz (demo)', 12450.75, 'TRY'),
    ('TR640001000000012345678901', 'Mehmet Demir (demo)', 3120.40, 'TRY')
ON CONFLICT (account_id) DO NOTHING;

INSERT INTO cards (card_last4, account_id, status) VALUES
    ('4321', 'TR330006100519786457841326', 'active'),
    ('9087', 'TR640001000000012345678901', 'active'),
    ('1122', 'TR640001000000012345678901', 'blocked')
ON CONFLICT (card_last4) DO NOTHING;

INSERT INTO transactions (account_id, occurred_at, description, amount) VALUES
    ('TR330006100519786457841326', '2026-08-03T14:22:00Z', 'Migros - market alışverişi', -245.50),
    ('TR330006100519786457841326', '2026-08-01T09:05:00Z', 'Maaş yatışı', 18500.00),
    ('TR330006100519786457841326', '2026-07-29T19:47:00Z', 'Elektrik faturası', -412.30),
    ('TR330006100519786457841326', '2026-07-27T11:10:00Z', 'ATM para çekme', -1000.00),
    ('TR640001000000012345678901', '2026-08-02T18:30:00Z', 'Netflix aboneliği', -129.99),
    ('TR640001000000012345678901', '2026-07-30T08:00:00Z', 'Kira ödemesi', -9500.00),
    ('TR640001000000012345678901', '2026-07-25T13:15:00Z', 'Serbest çalışma ödemesi', 6200.00)
ON CONFLICT (account_id, occurred_at, description) DO NOTHING;
