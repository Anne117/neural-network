#include "NetWork.h"
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>

struct data_info
{                          // структура для 26 букв
    vector<double> pixels; // пиксели к каждой букве (28х28)
    int letter;
};

data_NetWork ReadDataNetWork(string path)
{ // считываем конфиг
    data_NetWork data{};
    ifstream fin;
    fin.open(path);
    if (!fin.is_open())
    {
        cout << "Error reading the file " << path << endl;
        system("pause");
    }
    else
        cout << path << " loading...\n";
    string tmp;
    int L;
    while (!fin.eof())
    {
        fin >> tmp;
        if (tmp == "NetWork")
        {
            fin >> L;
            data.L = L;
            data.size.resize(L);
            for (int i = 0; i < L; i++)
            {
                fin >> data.size[i];
            }
        }
    }
    fin.close();
    return data;
}

vector<data_info> shuffle(vector<data_info> data)
{
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(data), std::end(data), rng);
    return data;
}

vector<data_info> ReadData(string path, const data_NetWork &data_NW, int &examples)
{ // считываем дату
    vector<data_info> data;
    ifstream fin;
    fin.open(path);
    if (!fin.is_open())
    {
        cout << "Error reading the file " << path << endl;
        system("pause");
    }
    else
        cout << path << " loading... \n";
    string tmp = "Examples";
    fin >> tmp;
    if (tmp == "Examples")
    {
        fin >> examples; // кол-во обучающих/тестовых экземпляров
        cout << "Examples: " << examples << endl;
        data.resize(examples); // выделяем память под дату
        for (int i = 0; i < examples; ++i)
            data[i].pixels.resize(data_NW.size[0]); // выделяем память под пиксели

        for (int i = 0; i < examples; ++i)
        {
            char letter;
            int letter_index;
            fin >> letter;                 // считываем букву
            letter_index = letter - 'A';   // преобразуем букву в индекс
            data[i].letter = letter_index; // создает строку из одиночного символа letter и присвоит ее переменной data[i].letter
            for (int j = 0; j < data_NW.size[0]; ++j)
            {
                fin >> data[i].pixels[j];
            }
        }
        fin.close();
        cout << "shuffling dataset... \n";

        // for (int i = 0; i < 5; ++i)
        // {
        //     cout << data[i].letter << " ";
        // }

        data = shuffle(data);

        // cout << endl;
        // for (int i = 0; i < 5; ++i)
        // {
        //     cout << data[i].letter << " ";
        // }
        // cout << endl;

        cout << "dataset loaded... \n";
        return data;
    }
    else
    {
        cout << "Error loading: " << path << endl;
        fin.close();
        exit(1);
    }
}

int main()
{
    NetWork NW{};
    data_NetWork NW_config;
    vector<data_info> data;
    double ra = 0, maxra = 0; // right answers за одну эпоху и макс значение за все прошедшие эпохи
    int right, predict;
    int epoch = 0;
    bool study, repeat = true;
    chrono::duration<double> time;

    NW_config = ReadDataNetWork("Config.txt"); // считываем данные из конфига
    NW.Init(NW_config);                        // инициализируем НС
    NW.PrintConfig();                          // выводим конфигурацию на экран

    while (repeat)
    {
        cout << "STUDY? (1/0)" << endl;
        cin >> study;
        if (study)
        {
            int examples;
            data = ReadData("train_104k.txt", NW_config, examples);
            auto begin = chrono::steady_clock::now();
            while (ra / examples * 100 < 100)
            { // обучение до 100% верных ответов
                ra = 0;
                auto t1 = chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i)
                {                                // идем по всем обучающим примерам
                    NW.SetInput(data[i].pixels); // подаем на вход пиксели
                    right = data[i].letter;      // правильная буква из даты
                    predict = NW.ForwardFeed();  // получаем предсказанную букву от НС
                    if (i % 10 == 0)
                    {
                        std::cout << i << "/" << examples << " > right: " << right << " | predict: " << predict << " ra : " << ra / i * 100 << "\r";
                    }
                    if (predict != right)
                    {                                                // если они не совпадают, то обучаем НС
                        NW.BackPropogation(right);                   // ищем дельты
                        NW.WeightsUpdater(0.15 * exp(-epoch / 20.)); // обновляем веса с помощью экспоненциального затухания
                    }
                    else
                        ra++;
                }
                std::cout << std::endl;
                auto t2 = chrono::steady_clock::now();
                time = t2 - t1;
                if (ra > maxra)
                    maxra = ra;
                cout << "ra: " << ra / examples * 100 << "\t"
                     << "maxra: " << maxra / examples * 100 << "\t"
                     << "epoch: " << epoch << "\tTIME: " << time.count() << endl;
                epoch++;
                if (epoch == 20) // ограничение на кол-во эпох
                    break;
            }
            auto end = chrono::steady_clock::now();
            time = end - begin;
            cout << "TIME: " << time.count() / 60. << " min" << endl;
            NW.SaveWeights();
        }
        else
        {
            NW.ReadWeights();
        }
        cout << "Test? (1/0)\n";
        bool test_flag;
        cin >> test_flag;
        if (test_flag)
        {
            int ex_tests;
            vector<data_info> data_test;
            data_test = ReadData("data_characters_test.txt", NW_config, ex_tests);
            ra = 0;
            for (int i = 0; i < ex_tests; ++i)
            {
                NW.SetInput(data_test[i].pixels);
                predict = NW.ForwardFeed();
                right = data_test[i].letter;
                std::cout << i << "/" << ex_tests << " > right: " << right << " | predict: " << predict << " ra : " << ra / i * 100 << endl;
                if (right == predict)
                    ra++;
            }
            cout << "RA: " << ra / ex_tests * 100 << endl;
        }
        cout << "Repeat? (1/0)\n";
        cin >> repeat;
    }
    system("pause");
    return 0;
}
