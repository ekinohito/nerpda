from deeppavlov import configs, build_model


def main():
    ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
    x = ner_model(['Чемпионат мира по кёрлингу пройдёт в Антананариву. На нем будет присутствовать Альберт Арбузович'])
    print(x)

    model = build_model("morpho_ru_syntagrus_bert", download=True, install=True)

    sentences = ["Чемпионат мира по кёрлингу пройдёт в Антананариву. На нем будет присутствовать в красивом платье Александра Абрамовна"]
    for parse in model(sentences):
        print(parse)



if __name__ == "__main__":
    main()
