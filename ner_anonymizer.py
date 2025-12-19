from ner import NatashaNER, RegexNER
from pda import PDAnonymizer
from typing import List, Tuple, Union


class NERAnonymizer:
    def __init__(self, mode: str = "mask"):
        """
        Инициализация анонимизатора
        
        Args:
            mode: Режим работы - "mask" для маскирования или "replace" для замены
        """
        if mode not in ["mask", "replace"]:
            raise ValueError("Режим должен быть 'mask' или 'replace'")
            
        self.mode = mode
        self.natasha_ner = NatashaNER()
        self.regex_ner = RegexNER()
        self.pd_anonymizer = PDAnonymizer()
    
    def anonymize(self, text: str) -> str:
        """
        Анонимизация текста с использованием NER
        
        Args:
            text: Исходный текст
            
        Returns:
            Анонимизированный текст
        """
        # Получаем все сущности из текста
        natasha_entities = self.natasha_ner.extract_names(text)
        regex_entities = self.regex_ner.extract_entities(text)
        
        # Объединяем все сущности и сортируем по начальной позиции
        all_entities = natasha_entities + regex_entities
        all_entities.sort(key=lambda x: x[2])  # Сортировка по start
        
        # Создаем список замен для применения
        replacements = []
        
        for entity in all_entities:
            tag, feats, start, end = entity
            original_text = text[start:end]
            
            if self.mode == "mask":
                # Режим маскирования
                replacement = "#" * len(original_text)
            else:
                # Режим замены
                replacement = self._get_replacement(tag, original_text, feats)
            
            replacements.append((start, end, replacement))
        
        # Применяем замены в обратном порядке, чтобы не сбить индексы
        result_text = text
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            result_text = result_text[:start] + replacement + result_text[end:]
        
        return result_text
    
    def _get_replacement(self, tag: str, original_text: str, feats: str) -> str:
        """
        Получение замены для сущности в режиме replace
        
        Args:
            tag: Тип сущности
            original_text: Оригинальный текст
            feats: Морфологические признаки
            
        Returns:
            Замененный текст
        """
        if tag == 'NAME':
            return self.pd_anonymizer.anonymize_name(original_text)
        elif tag == 'SURNAME':
            return self.pd_anonymizer.anonymize_last_name(original_text)
        elif tag == 'EMAIL':
            return self.pd_anonymizer.anonymize_email(original_text)
        elif tag == 'CARD_NUMBER':
            return self.pd_anonymizer.anonymize_card(original_text)
        elif tag == 'OGRNIP':
            return self.pd_anonymizer.anonymize_ogrnip(original_text)
        elif tag == 'IP_ADDRESS':
            return self.pd_anonymizer.anonymize_ip(original_text)
        elif tag == 'BIRTH_CERT':
            return self.pd_anonymizer.anonymize_birth_cert(original_text)
        elif tag == 'KLADR':
            return self.pd_anonymizer.anonymize_kladr(original_text)
        else:
            # Для неизвестных типов сущностей используем маскирование
            return "#" * len(original_text)
    
    def extract_and_anonymize(self, text: str) -> Tuple[str, List[List[Union[str, int]]]]:
        """
        Извлечение сущностей и анонимизация текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Кортеж из анонимизированного текста и списка найденных сущностей
        """
        # Получаем все сущности
        natasha_entities = self.natasha_ner.extract_names(text)
        regex_entities = self.regex_ner.extract_entities(text)
        
        # Объединяем и сортируем сущности
        all_entities = natasha_entities + regex_entities
        all_entities.sort(key=lambda x: x[2])  # Сортировка по start
        
        # Анонимизируем текст
        anonymized_text = self.anonymize(text)
        
        return anonymized_text, all_entities


# Пример использования
if __name__ == "__main__":
    # Создаем анонимизатор в режиме маскирования
    mask_anonymizer = NERAnonymizer(mode="mask")
    
    # Создаем анонимизатор в режиме замены
    replace_anonymizer = NERAnonymizer(mode="replace")
    
    # Тестовый текст
    test_text = """
    Регистрация ИП: Виноградов Никита Олегович, ОГРНИП 255547853460739, ИНН 7801920588. 
    Адрес по КЛАДР: 27 328 191 574. Email для связи: никита764@list.ru. 
    Зарплатный проект: сотрудник Виноградов Никита Олегович, ИНН 7801920588, карта 1558 5701 8528 1280. 
    Код КЛАДР для налоговой: 27 328 191 574. Клиент Виноградов Никита Олегович предоставил документы: 
    паспорт 3255 141253, ИНН 7801920588, свидетельство о рождении X-VI №901709. 
    Регистрация в системе: пользователь Виноградов Никита Олегович, email никита764@list.ru, 
    ИНН 7801920588, ОГРНИП 255547853460739 (если ИП). Для верификации необходимы данные: 
    Виноградов Никита Олегович, ИНН 7801920588, паспорт 3255 141253, email никита764@list.ru, 
    IP-адрес 76.79.127.227. Данные предпринимателя: Виноградов Никита Олегович, 
    ОГРНИП 255547853460739. Банковская карта 1558 5701 8528 1280. ИНН 7801920588. 
    Адрес по КЛАДР: 27 328 191 574.
    """
    
    # Анонимизация в режиме маскирования
    masked_text, entities = mask_anonymizer.extract_and_anonymize(test_text)
    print("Режим маскирования:")
    print(masked_text)
    print("\nНайденные сущности:", entities)
    
    # Анонимизация в режиме замены
    replaced_text, entities = replace_anonymizer.extract_and_anonymize(test_text)
    print("\n\nРежим замены:")
    print(replaced_text)
    print("\nНайденные сущности:", entities)