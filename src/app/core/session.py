"""Oturum açmış müşteri kavramı.

Neden var: sistem bu ana kadar her konuşmayı anonim başlatıyordu ve bir kart
işlemi için "kartınızın son 4 hanesi nedir?" diye soruyordu. Bu, telefon
bankacılığı refleksi — orada karşıdaki kişinin kim olduğu bilinmiyor, o yüzden
kimlik kanıtı isteniyor.

Uygulama içi bir asistanda bu saçma. Kullanıcı zaten giriş yapmış; banka onun
kim olduğunu, kaç kartı olduğunu ve numaralarını zaten biliyor. Elindeki
bilgiyi kullanmayıp kullanıcıdan istemek, ürünün kendi verisini görmezden
gelmesi demek.

Bu modülden sonra doğru soru değişiyor:

    önce:  "Kartınızın son 4 hanesi nedir?"          (kimlik kanıtı)
    sonra: "4321 ve 9087 ile biten iki kartınız var.  (netleştirme)
            Hangisini bloke edeyim?"

Demo sınırı, açıkça: burada gerçek bir kimlik doğrulama yok. Sabit bir müşteri
kimliği, bir oturum belirtecinden okunuyormuş gibi davranıyor. Gerçek bir
üründe bu değer JWT'den ya da oturum deposundan gelir; buradaki tek amaç
"asistan kullanıcıyı tanır" varsayımını sisteme sokmak.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class CustomerSession:
    """Konuşmayı yapan müşteri. Gerçek üründe oturum belirtecinden gelir."""

    customer_name: str
    account_id: str

    @property
    def masked_account(self) -> str:
        """IBAN'ın ekranda gösterilebilir hâli — son 4 hane dışında maskeli.

        Guardrail zaten uzun sayı dizilerini redakte ediyor (ADR-006); burada
        kaynağında maskelemek, o katmana iş bırakmamak için.
        """
        return f"{self.account_id[:6]}...{self.account_id[-4:]}" if self.account_id else ""


def get_session(settings: Settings) -> CustomerSession:
    """Aktif oturum. Demoda tek müşteri, ayarlardan okunuyor."""
    return CustomerSession(
        customer_name=settings.demo_customer_name,
        account_id=settings.demo_account_id,
    )
