"""Kullanıcıya dönük hata mesajları ve kart netleştirme sorusu.

Buradaki iddia şu: bir kullanıcı ekranda asla ham bir hata kodu görmemeli.
`CARD_NOT_FOUND` bir mühendis için bilgi, bir kullanıcı için gürültü — ne
olduğunu da söylemiyor, ne yapılması gerektiğini de.

Demoyu ilk açan biri rastgele bir kart numarası girdiğinde "sistem patladı"
izlenimi almamalı; ne gireceğini sistemin kendisi söylemeli.
"""

from __future__ import annotations

import pytest

from agents.workers.tool_agent import (
    TOOL_ERROR_MESSAGES,
    build_card_prompt,
    humanize_tool_error,
)


class TestHumanizeToolError:
    @pytest.mark.parametrize("code", sorted(TOOL_ERROR_MESSAGES))
    def test_ham_kod_mesaja_sizmiyor(self, code: str) -> None:
        """En temel kural: hata kodunun kendisi çıktıda görünmemeli."""
        assert code not in humanize_tool_error(code, cards="4321", account="TR33")

    def test_kart_bulunamadi_kayitli_kartlari_soyluyor(self) -> None:
        """Kullanıcı ne gireceğini bilmiyor; mesaj bunu söylemek zorunda."""
        message = humanize_tool_error("CARD_NOT_FOUND", cards="4321 ile biten, 9087 ile biten")
        assert "4321" in message and "9087" in message

    def test_bilinmeyen_kod_genel_mesaja_dusuyor(self) -> None:
        message = humanize_tool_error("SOME_NEW_ERROR_NOBODY_MAPPED")
        assert "SOME_NEW_ERROR" not in message
        assert message.strip()

    def test_servis_hatasi_bir_sonraki_adimi_oneriyor(self) -> None:
        """Hata mesajı çıkmaz sokak olmamalı — ne yapılacağını söylemeli."""
        message = humanize_tool_error("BANKING_SERVICE_UNAVAILABLE")
        assert "temsilci" in message.lower()

    def test_bos_kart_listesiyle_de_patlamiyor(self) -> None:
        assert humanize_tool_error("CARD_NOT_FOUND").strip()

    @pytest.mark.parametrize("code", sorted(TOOL_ERROR_MESSAGES))
    def test_mesajlar_turkce_ve_dolu(self, code: str) -> None:
        message = humanize_tool_error(code, cards="4321", account="TR33")
        assert len(message) > 20
        assert message[0].isupper()


class TestBuildCardPrompt:
    def test_tek_kartta_secim_sorulmuyor(self) -> None:
        """Bildiğimiz bir şeyi sormak, kendi verimizi görmezden gelmek demek."""
        prompt = build_card_prompt([{"last4": "4321", "status": "active"}])
        assert "4321" in prompt
        assert "Hangisini" not in prompt

    def test_coklu_kartta_hepsi_listeleniyor(self) -> None:
        prompt = build_card_prompt(
            [{"last4": "4321", "status": "active"}, {"last4": "9087", "status": "active"}]
        )
        assert "4321" in prompt and "9087" in prompt
        assert "Hangisini" in prompt

    def test_bloke_kartlar_secenek_olarak_sunulmuyor(self) -> None:
        prompt = build_card_prompt(
            [{"last4": "4321", "status": "active"}, {"last4": "1122", "status": "blocked"}]
        )
        assert "4321" in prompt
        assert "1122" not in prompt

    def test_aktif_kart_yoksa_insana_yonlendiriyor(self) -> None:
        prompt = build_card_prompt([{"last4": "1122", "status": "blocked"}])
        assert "temsilci" in prompt.lower()

    def test_hic_kart_yoksa_patlamiyor(self) -> None:
        assert build_card_prompt([]).strip()

    def test_son_4_hane_sorusu_artik_sorulmuyor(self) -> None:
        """Bu, değişikliğin asıl amacı: kimlik kanıtı değil netleştirme.

        "Kartınızın son 4 hanesi nedir?" telefon bankacılığı refleksi —
        karşıdakinin kim olduğu bilinmediğinde mantıklı. Uygulama içi bir
        asistanda kullanıcı zaten giriş yapmış durumda.
        """
        prompt = build_card_prompt(
            [{"last4": "4321", "status": "active"}, {"last4": "9087", "status": "active"}]
        )
        assert "son 4" not in prompt.lower()
