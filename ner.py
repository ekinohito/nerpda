from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)
import re


class NatashaNER:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)

    def extract_names(self, text):
        doc = Doc(text)
        
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)
        
        result = []
        
        if doc.spans is not None:
            for span in doc.spans:
                if span.type == PER:
                    span.normalize(self.morph_vocab)
                    span.extract_fact(self.names_extractor)
                    
                    # Zip tokens and slots
                    if span.fact and span.fact.slots:
                        for token, slot in zip(span.tokens, span.fact.slots):
                            # Determine tag based on slot key
                            if slot.key == 'last':
                                tag = 'SURNAME'
                            elif slot.key == 'first':
                                tag = 'NAME'
                            else:
                                continue  # Skip other slots
                            
                            # Get morphological features
                            feats = token.feats
                            
                            # Get start and end positions
                            start = token.start
                            end = token.stop
                            
                            result.append([tag, str(feats), start, end])
        
        return result


class RegexNER:
    def __init__(self):
        # Регулярное выражение для номера банковской карты (Visa, Mastercard, Maestro, Мир)
        # Формат: 13-19 цифр, с проверкой по алгоритму Луна
        self.card_pattern = re.compile(r'\b(?:\d[ -]*?){13,19}\b')
        
        # Регулярное выражение для ОГРНИП (15 цифр)
        # Формат: XXXXXXXXXXXXXXX, где последняя цифра - контрольное число (N14 mod 11 mod 10)
        self.ogrnip_pattern = re.compile(r'\b\d{15}\b')
        
        # Регулярное выражение для IP-адреса
        self.ip_pattern = re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b')
        
        # Регулярное выражение для email
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        
        # Регулярное выражение для номера свидетельства о рождении
        # Формат: XII-АБ №123456 (римские цифры, русские буквы, номер)
        self.birth_cert_pattern = re.compile(r'\b(?:[IVXLCDM]+-[А-ЯЁ]{2}\s*№\d{6})\b', re.IGNORECASE)
        
        # Регулярное выражение для кода КЛАДР
        # Формат: XX YY ZZZ QQ, где XX - код региона, YY - код района, ZZZ - код города/населенного пункта, QQ - код улицы
        self.kladr_pattern = re.compile(r'\b\d{2}\s*\d{2}\s*\d{3}\s*\d{2}\b')
    
    def luhn_algorithm(self, card_number):
        """Проверка номера карты по алгоритму Луна"""
        card_number = card_number.replace(' ', '').replace('-', '')
        if not card_number.isdigit():
            return False
        
        total = 0
        for i, digit in enumerate(card_number):
            digit = int(digit)
            if i % 2 == len(card_number) % 2:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        
        return total % 10 == 0
    
    def validate_ogrnip(self, ogrnip):
        """Проверка ОГРНИП"""
        if len(ogrnip) != 15 or not ogrnip.isdigit():
            return False
        
        # Последняя цифра - контрольное число
        main_part = int(ogrnip[:14])
        control_digit = int(ogrnip[14])
        
        # Вычисляем контрольное число
        calculated_control = main_part % 11 % 10
        
        return calculated_control == control_digit
    
    def extract_entities(self, text):
        result = []
        
        # Поиск номеров банковских карт
        for match in self.card_pattern.finditer(text):
            card_number = match.group().replace(' ', '').replace('-', '')
            if self.luhn_algorithm(card_number):
                result.append(['CARD_NUMBER', 'valid_card', match.start(), match.end()])
        
        # Поиск ОГРНИП
        for match in self.ogrnip_pattern.finditer(text):
            if self.validate_ogrnip(match.group()):
                result.append(['OGRNIP', 'valid_ogrnip', match.start(), match.end()])
        
        # Поиск IP-адресов
        for match in self.ip_pattern.finditer(text):
            result.append(['IP_ADDRESS', 'valid_ip', match.start(), match.end()])
        
        # Поиск email
        for match in self.email_pattern.finditer(text):
            result.append(['EMAIL', 'valid_email', match.start(), match.end()])
        
        # Поиск номеров свидетельств о рождении
        for match in self.birth_cert_pattern.finditer(text):
            result.append(['BIRTH_CERT', 'valid_birth_cert', match.start(), match.end()])
        
        # Поиск кодов КЛАДР
        for match in self.kladr_pattern.finditer(text):
            result.append(['KLADR', 'valid_kladr', match.start(), match.end()])
        
        return result


# Example usage
if __name__ == "__main__":
    # Test NatashaNER
    ner = NatashaNER()
    # Test RegexNER
    regex_ner = RegexNER()
    text = """
    Регистрация ИП: Виноградов Никита Олегович, ОГРНИП 255547853460739, ИНН 7801920588. Адрес по КЛАДР: 27 328 191 574. Email для связи: никита764@list.ru. Зарплатный проект: сотрудник Виноградов Никита Олегович, ИНН 7801920588, карта 1558 5701 8528 1280. Код КЛАДР для налоговой: 27 328 191 574. Клиент Виноградов Никита Олегович предоставил документы: паспорт 3255 141253, ИНН 7801920588, свидетельство о рождении X-VI №901709. Регистрация в системе: пользователь Виноградов Никита Олегович, email никита764@list.ru, ИНН 7801920588, ОГРНИП 255547853460739 (если ИП). Для верификации необходимы данные: Виноградов Никита Олегович, ИНН 7801920588, паспорт 3255 141253, email никита764@list.ru, IP-адрес 76.79.127.227. Данные предпринимателя: Виноградов Никита Олегович, ОГРНИП 255547853460739. Банковская карта 1558 5701 8528 1280. ИНН 7801920588. Адрес по КЛАДР: 27 328 191 574.
    """
    names = ner.extract_names(text)
    entities = regex_ner.extract_entities(text)
    print("Entities:", names, entities)