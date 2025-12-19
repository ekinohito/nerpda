from ner_anonymizer import NERAnonymizer


def main():
    # Создаем анонимизаторы в двух режимах
    mask_anonymizer = NERAnonymizer(mode="mask")
    replace_anonymizer = NERAnonymizer(mode="replace")
    
    print("=== Система анонимизации текста ===")
    print("Два режима анонимизации:")
    print("1. Маскирование (заменяет персональные данные на #)")
    print("2. Замена (заменяет персональные данные на сгенерированные)")
    print("Введите 'exit' для выхода из программы")
    print("=" * 40)
    
    while True:
        # Получаем текст от пользователя
        text = input("\nВведите текст для анонимизации: ")
        
        # Проверяем условие выхода
        if text.lower() == 'exit':
            print("Завершение работы программы...")
            break
        
        if not text.strip():
            print("Ошибка: пустой текст. Пожалуйста, введите текст.")
            continue
        
        print("\nИсходный текст:")
        print(text)
        
        # Анонимизация в режиме маскирования
        masked_text, mask_entities = mask_anonymizer.extract_and_anonymize(text)
        print("\n1. Режим маскирования:")
        print(masked_text)
        print(f"Найдено сущностей: {len(mask_entities)}")
        
        # Анонимизация в режиме замены
        replaced_text, replace_entities = replace_anonymizer.extract_and_anonymize(text)
        print("\n2. Режим замены:")
        print(replaced_text)
        print(f"Найдено сущностей: {len(replace_entities)}")
        
        print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
