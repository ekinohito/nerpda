import random
import re
from faker import Faker
import pymorphy3
from typing import Dict, Any

class PDAnonymizer:
    def __init__(self):
        self.faker = Faker('ru_RU')
        self.morph = pymorphy3.MorphAnalyzer()
    
    def _get_morph_features(self, word: str) -> Dict[str, Any]:
        """Получить морфологические характеристики слова"""
        if not isinstance(word, str) or not word.strip():
            return {'gender': 'masc', 'number': 'sing', 'case': 'nomn'}
            
        try:
            parsed = self.morph.parse(word)[0]
            return {
                'gender': parsed.tag.gender if parsed.tag.gender else 'masc',
                'number': parsed.tag.number if parsed.tag.number else 'sing',
                'case': parsed.tag.case if parsed.tag.case else 'nomn'
            }
        except Exception:
            return {'gender': 'masc', 'number': 'sing', 'case': 'nomn'}
    
    def _inflect_word(self, word: str, target_features: Dict[str, Any]) -> str:
        """Просклонять слово по заданным характеристикам"""
        if not isinstance(word, str) or not word.strip():
            return word
            
        try:
            parsed = self.morph.parse(word)[0]
            
            # Создаем набор тегов для склонения
            target_set = set()
            if target_features.get('gender'):
                target_set.add(target_features['gender'])
            if target_features.get('number'):
                target_set.add(target_features['number'])
            if target_features.get('case'):
                target_set.add(target_features['case'])
            
            if target_set:
                inflected = parsed.inflect(target_set)
                if inflected:
                    return inflected.word
                else:
                    return word
            else:
                return word
                
        except Exception:
            return word
    
    def anonymize_name(self, name: str, features: dict | None = None) -> str:
        """Обезличивание имени с сохранением морфологии"""
        try:
            # Проверка на корректность данных
            if not isinstance(name, str) or not name.strip():
                return "Некорректно введены данные"
            
            # Проверка на цифры и специальные символы в имени
            if re.search(r'[\d@#$%^&*()_+=\[\]{}|;:",.<>?/\\]', name):
                return "Некорректно введены данные"
            
            # Получаем морфологические характеристики оригинала
            original_features = features or self._get_morph_features(name)
            
            # Генерируем случайное имя того же рода с помощью Faker
            if original_features.get('gender') == 'femn':
                new_name = self.faker.first_name_female()
            else:
                new_name = self.faker.first_name_male()
            
            # Склоняем новое имя по характеристикам оригинала
            anonymized_name = self._inflect_word(new_name, original_features)
            
            # Возвращаем результат
            return anonymized_name.capitalize()
            
        except Exception:
            return "Некорректно введены данные"
    
    def anonymize_last_name(self, last_name: str, features: dict | None = None) -> str:
        """Обезличивание фамилии с сохранением морфологии"""
        try:
            # Проверка на корректность данных
            if not isinstance(last_name, str) or not last_name.strip():
                return "Некорректно введены данные"
            
            # Проверка на цифры и специальные символы в фамилии
            if re.search(r'[\d@#$%^&*()_+=\[\]{}|;:",.<>?/\\]', last_name):
                return "Некорректно введены данные"
            
            # Получаем морфологические характеристики оригинала
            original_features = features or self._get_morph_features(last_name)
            
            # Генерируем случайную фамилию того же рода с помощью Faker
            if original_features.get('gender') == 'femn':
                new_last_name = self.faker.last_name_female()
            else:
                new_last_name = self.faker.last_name_male()
            
            # Склоняем новую фамилию по характеристикам оригинала
            anonymized_last_name = self._inflect_word(new_last_name, original_features)
            
            # Возвращаем результат
            return anonymized_last_name.capitalize()
            
        except Exception:
            return "Некорректно введены данные"
    
    def anonymize_email(self, email: str) -> str:
        """Обезличивание email с сохранением домена"""
        try:
            # Проверка на корректность данных
            if not isinstance(email, str) or not email.strip():
                return "Некорректно введены данные"
            
            # Проверка формата email
            if '@' not in email:
                return "Некорректно введены данные"
            
            local_part, domain = email.split('@', 1)
            new_local = self.faker.user_name()
            return f"{new_local}@{domain}"
            
        except Exception:
            return "Некорректно введены данные"
    
    def _calc_inn_control(self, digits: str, weights: list[int]) -> str:
        s = sum(int(d) * w for d, w in zip(digits, weights))
        return str(s % 11 % 10)

    def anonymize_inn(self, inn: str) -> str:
        """Обезличивание ИНН с сохранением первых 4 цифр и контрольного разряда"""
        try:
            if not isinstance(inn, str) or not inn.isdigit():
                return "Некорректно введены данные"

            if len(inn) == 10:
                base = ''.join(str(random.randint(0, 9)) for _ in range(9))
                control = self._calc_inn_control(
                    base, [2, 4, 10, 3, 5, 9, 4, 6, 8]
                )
                return base + control

            if len(inn) == 12:
                base = ''.join(str(random.randint(0, 9)) for _ in range(10))
                c1 = self._calc_inn_control(
                    base[:10], [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
                )
                c2 = self._calc_inn_control(
                    base[:11] + c1,
                    [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
                )
                return base[:10] + c1 + c2

            return "Некорректно введены данные"
        except Exception:
            return "Некорректно введены данные"
    
    def anonymize_phone(self, phone: str) -> str:
        """Обезличивание номера телефона с сохранением формата"""
        try:
            # Проверка на корректность данных
            if not isinstance(phone, str) or not phone.strip():
                return "Некорректно введены данные"
            
            # Строгая проверка: допустимы только цифры, +, (), - и пробелы
            if not re.match(r'^[\d\+\-\(\)\s]+$', phone):
                return "Некорректно введены данные"
            
            digits = re.findall(r'\d', phone)
            
            # Проверка минимальной длины телефона
            if len(digits) < 7:
                return "Некорректно введены данные"
                
            preserved_digits = digits[:4]
            new_digits = [str(random.randint(0, 9)) for _ in range(len(digits) - 4)]
            
            result = []
            digit_index = 0
            new_all_digits = preserved_digits + new_digits
            
            for char in phone:
                if char.isdigit():
                    if digit_index < len(new_all_digits):
                        result.append(new_all_digits[digit_index])
                        digit_index += 1
                    else:
                        result.append(char)
                else:
                    result.append(char)
            
            return ''.join(result)
            
        except Exception:
            return "Некорректно введены данные"
    
    def anonymize_passport(self, passport: str) -> str:
        try:
            m = re.fullmatch(r'(\d{2})\s?(\d{2})\s?(\d{6})', passport)
            if not m:
                return "Некорректно введены данные"

            series = ''.join(str(random.randint(0, 9)) for _ in range(4))
            number = ''.join(str(random.randint(0, 9)) for _ in range(6))
            return f"{series[:2]} {series[2:]} {number}"
        except Exception:
            return "Некорректно введены данные"
    
    def _luhn_checksum(self, digits: str) -> str:
        s = 0
        for i, d in enumerate(reversed(digits)):
            n = int(d)
            if i % 2 == 0:
                n *= 2
                if n > 9:
                    n -= 9
            s += n
        return str((10 - s % 10) % 10)
    
    def anonymize_card(self, card: str) -> str:
        try:
            digits = re.sub(r'\D', '', card)
            if not (13 <= len(digits) <= 19):
                return "Некорректно введены данные"

            body = ''.join(str(random.randint(0, 9)) for _ in range(len(digits) - 1))
            check = self._luhn_checksum(body)
            new_digits = body + check

            result = []
            i = 0
            for ch in card:
                if ch.isdigit():
                    result.append(new_digits[i])
                    i += 1
                else:
                    result.append(ch)
            return ''.join(result)
        except Exception:
            return "Некорректно введены данные"
    
    def anonymize_ogrnip(self, ogrnip: str) -> str:
        try:
            if not ogrnip.isdigit() or len(ogrnip) != 15:
                return "Некорректно введены данные"

            base = ''.join(str(random.randint(0, 9)) for _ in range(14))
            control = str(int(base) % 13 % 10)
            return base + control
        except Exception:
            return "Некорректно введены данные"
    

    def anonymize_ip(self, ip: str) -> str:
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return "Некорректно введены данные"

            new_parts = [str(random.randint(1, 254)) for _ in range(4)]
            return '.'.join(new_parts)
        except Exception:
            return "Некорректно введены данные"
 
    def anonymize_birth_cert(self, cert: str) -> str:
        try:
            # Проверяем формат с римскими цифрами
            m = re.fullmatch(r'([IVXLCDM]+)-([А-ЯЁ]{2})\s*№(\d{6})', cert, re.IGNORECASE)
            if not m:
                return "Некорректно введены данные"

            # Генерируем новые римские цифры (от 1 до 20)
            roman_numeral = self._int_to_roman(random.randint(1, 20))
            letters = ''.join(random.choice('АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ') for _ in range(2))
            number = ''.join(str(random.randint(0, 9)) for _ in range(6))
            return f"{roman_numeral}-{letters} №{number}"
        except Exception:
            return "Некорректно введены данные"
    
    def _int_to_roman(self, num: int) -> str:
        """Преобразование целого числа в римские цифры"""
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syb[i]
                num -= val[i]
            i += 1
        return roman_num
    
    def anonymize_kladr(self, kladr: str) -> str:
        try:
            # Удаляем все пробелы и проверяем формат
            digits = re.sub(r'\s', '', kladr)
            if not digits.isdigit() or len(digits) < 9:
                return "Некорректно введены данные"

            # Генерируем новый код КЛАДР той же длины
            new_digits = ''.join(str(random.randint(0, 9)) for _ in range(len(digits)))
            
            # Восстанавливаем исходный формат (с пробелами)
            result = new_digits
            if ' ' in kladr:
                # Восстанавливаем пробелы на тех же позициях
                for i, char in enumerate(kladr):
                    if char == ' ' and i < len(result):
                        result = result[:i] + ' ' + result[i:]
            
            return result
        except Exception:
            return "Некорректно введены данные"

