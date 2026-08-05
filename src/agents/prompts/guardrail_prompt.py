"""Guardrail düğümü bir yanıtı engellediğinde/redakte ettiğinde kullanıcıya gösterilen metinler."""

FINANCIAL_ADVICE_DISCLAIMER = (
    "Bu konuda kişiye özel yatırım tavsiyesi veremiyorum. "
    "Genel ürün bilgisi için bir bankacılık uzmanına yönlendirebilirim."
)

ITERATION_LIMIT_MESSAGE = (
    "Talebinizi tamamlamak için gereken adım sayısını aştım ve hatalı bir işlem yapmamak adına "
    "burada durdum. Bir müşteri temsilcisine aktarıyorum."
)

NO_DRAFT_FALLBACK_MESSAGE = (
    "Bu talebi şu an yanıtlayamadım. Sorunuzu farklı bir şekilde iletebilir ya da bir müşteri "
    "temsilcisiyle görüşmek isteyip istemediğinizi belirtebilirsiniz."
)
