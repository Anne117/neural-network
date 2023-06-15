#include "NetWork.h"
#include <iostream>

void NetWork::Init(data_NetWork data)
{
	std::cout << "Set actFunc pls\n1 - sigmoid \n2 - ReLU \n3 - th(x) \n";
	int tmp;
	std::cin >> tmp;

  switch (tmp)
  {
    case 1:
      actFunc = std::make_unique<SigmoidActivation>();
      break;
    case 2:
      actFunc = std::make_unique<ReLUActivation>();
      break;
    case 3: 
      actFunc = std::make_unique<ThxActivation>();
      break;
    default:
      throw std::runtime_error("Error read actFunc");
      break;
  }
	srand(time(NULL)); // инициализируем генератор случайных чисел
	L = data.L;		   // запоминаем сколько слоев в нашей НС
	size.resize(L);	   // выделяем память под массив сайз и заполняем его значениями
	for (int i = 0; i < L; i++)
		size[i] = data.size[i];

	weights.resize(L - 1); // выделяем память под матрицу весов
	bios.resize(L - 1);	   // выделяем память под нейроны смещения
	for (int i = 0; i < L - 1; i++)
	{
		weights[i].Init(size[i + 1], size[i]);
		bios[i].resize(size[i + 1]);
		weights[i].Rand(); // заполняем матрицу весов рандомными числами
		for (int j = 0; j < size[i + 1]; j++)
		{
			bios[i][j] = ((rand() % 50)) * 0.06 / (size[i] + 15); // заполняем нейроны смещения рандомными весами по этой зависимости
		}
	}
	neurons_val.resize(L);
	neurons_err.resize(L);
	for (int i = 0; i < L; i++)
	{
		neurons_val[i].resize(size[i]);
		neurons_err[i].resize(size[i]);
	} // выделяем память под все остальные массивы, включаня нейроны смещения, и заполняем их единицами
	neurons_bios_val.resize(L - 1);
	for (int i = 0; i < L - 1; i++)
		neurons_bios_val[i] = 1;
}
void NetWork::PrintConfig()
{
	cout << "***********************************************************\n";
	cout << "NetWork has " << L << " layers\nSIZE[]: ";
	for (int i = 0; i < L; i++)
	{
		cout << size[i] << " ";
	} // выводим на экран кол-во слоев в нашей НС и массив сайз на экран
	cout << "\n***********************************************************\n\n";
}
void NetWork::SetInput(vector<double> &values)
{
	for (int i = 0; i < size[0]; i++)
	{
		neurons_val[0][i] = values[i];
	} // подаем на вход данные
}
void NetWork::Softmax(vector<double> &values)
{
	double maxVal = *max_element(values.begin(), values.end());
	double sumExp = 0.0;
	for (int i = 0; i < size[L - 1]; i++)
	{
		values[i] = exp(values[i] - maxVal);
		sumExp += values[i];
	}
	for (int i = 0; i < size[L - 1]; i++)
	{
		values[i] /= sumExp;
	}
}

double NetWork::ForwardFeed()
{ // функция прямого распространения
	for (int k = 1; k < L; ++k)
	{
		Matrix::Multi(weights[k - 1], neurons_val[k - 1], size[k - 1], neurons_val[k]); // умножаем матрицу весов на размер и значения нейронов
		Matrix::SumVector(neurons_val[k], bios[k - 1], size[k]);						// суммироуем предыдущий вектор с биосом
		actFunc->use(neurons_val[k], size[k]);											// применяет ФА на значения нейронов текущего слоя к
	}																					// ФА применяется к выходу каждого нейрона текущего слоя, т е ФА применяется на всех скрытых слоях и softmax на выходном слое сети
	Softmax(neurons_val[L - 1]);														// Применение функции Softmax к последнему слою
	int pred = SearchMaxIndex(neurons_val[L - 1]);
	return pred; // возвращаем нашу букву в качестве ответа от НС при обучении
}
int NetWork::SearchMaxIndex(vector<double> &value)
{ // чтобы получить ответ от НС эта функция принимает на вход вектор значений и ищет в нем макс. элемент и возвращает индекс этого элемента
	// argmax - возвращает индекс максимального элемента в векторе
	int maxIndex = 0;
	double maxVal = value[0];
	for (int i = 0; i < value.size(); i++)
	{
		if (value[i] > maxVal)
		{
			maxVal = value[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}
void NetWork::PrintValues(int L)
{
	for (int j = 0; j < size[L]; j++)
	{
		cout << j << " " << neurons_val[L][j] << endl;
	}
} // выводим на экран индекс и значение нейрона на данном этапе

void NetWork::BackPropogation(int expect)
{
	// метод ошибки/потерь, на входе у него правильная буква
	// вычисление ошибки для последнего слоя с использованием softmax
	for (int i = 0; i < size[L - 1]; i++)
	{					 // в цикле считаем дельту для выходных нейронов
		if (i != expect) // преобразование номера класса i в соответствующий аски-код заглавной латинской буквы затем мы сравниваем его с первым символом строки expect
			neurons_err[L - 1][i] = -neurons_val[L - 1][i] * actFunc->useDer(neurons_val[L - 1][i]);
		else
			neurons_err[L - 1][i] = (1.0 - neurons_val[L - 1][i]) * actFunc->useDer(neurons_val[L - 1][i]);
	}
	// обновление ошибок для внутренних слоев
	for (int k = L - 2; k > 0; k--)
	{ // считаем дельту для скрытых нейронов
		Matrix::Multi_T(weights[k], neurons_err[k + 1], size[k + 1], neurons_err[k]);
		for (int j = 0; j < size[k]; j++)
			neurons_err[k][j] *= actFunc->useDer(neurons_val[k][j]); // умножаем на производную ФА
	}
}

void NetWork::WeightsUpdater(double lr)
{ // обновление весов и смещений для всех слоев но
	// для последнего слоя учитываем разницу между значениями ошибки и значениями после применения функции softmax.
	// это позволяет учесть влияние функции softmax на обновление весов и смещений последнего слоя
	for (int i = 0; i < L - 1; ++i)
	{
		for (int j = 0; j < size[i + 1]; ++j)
		{
			for (int k = 0; k < size[i]; ++k)
			{
				weights[i](j, k) += neurons_val[i][k] * neurons_err[i + 1][j] * lr;
			}
		}
		if (i == L - 2)
		{
			for (int j = 0; j < size[i + 1]; ++j)
			{
				for (int k = 0; k < size[i]; ++k)
				{
					weights[i](j, k) += neurons_val[i][k] * (neurons_err[i + 1][j] - neurons_val[L - 1][j]) * lr;
				}
			}
		}
	}
	for (int i = 0; i < L - 1; i++)
	{
		for (int k = 0; k < size[i + 1]; k++)
		{
			bios[i][k] += neurons_err[i + 1][k] * lr;
		}
		if (i == L - 2)
		{
			for (int k = 0; k < size[i + 1]; k++)
			{
				bios[i][k] += (neurons_err[i + 1][k] - neurons_val[L - 1][k]) * lr;
			}
		}
	}
}

void NetWork::SaveWeights()
{
	ofstream fout;
	fout.open("Weights.txt");
	if (!fout.is_open())
	{
		cout << "Error opening the file";
		system("pause");
		return;
	}

	// открываем файл и записываем все веса в файл
	for (int i = 0; i < L - 1; ++i)
	{
		for (int j = 0; j < size[i + 1]; ++j)
		{
			for (int k = 0; k < size[i]; ++k)
			{
				fout << weights[i](j, k) << " ";
			}
			fout << "\n"; // добавляем новый символ строки после каждой строки весов
		}
	}
	for (int i = 0; i < L - 1; ++i)
	{
		for (int j = 0; j < size[i + 1]; ++j)
		{
			fout << bios[i][j] << " ";
		}
		fout << "\n"; // добавляем новый символ строки после каждого набора смещений
	}

	cout << "Weights saved\n";
	fout.close();
}
void NetWork::ReadWeights()
{
	ifstream fin;
	fin.open("Weights.txt");
	if (!fin.is_open())
	{
		cout << "Error opening the file";
		system("pause");
		return;
	}

	// читаем веса нейронов и нейронов смещения из файла
	for (int i = 0; i < L - 1; ++i)
	{
		for (int j = 0; j < size[i + 1]; ++j)
		{
			for (int k = 0; k < size[i]; ++k)
			{
				fin >> weights[i](j, k);
			}
		}
	}
	for (int i = 0; i < L - 1; ++i)
	{
		for (int j = 0; j < size[i + 1]; ++j)
		{
			fin >> bios[i][j];
		}
	}

	cout << "Weights read\n";
	fin.close();
}

NetWork::~NetWork()
{

}
