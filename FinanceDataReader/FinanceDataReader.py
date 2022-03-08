import FinanceDataReader as fdr
import os

class FinanceData:
    def __init__(self, market="KRX"):
        self.stock_list = fdr.StockListing(market)

    def get_stock_list(self):
        stock_list = self.stock_list
        stock_list = stock_list.loc[:, ["Symbol", "Market", "Name"]]
        stock_list = stock_list.sort_values(by="Symbol")

        drop_market_index = stock_list[(stock_list["Market"] != "KOSPI") & (stock_list["Market"] != "KOSDAQ")].index
        stock_list = stock_list.drop(drop_market_index)
        return stock_list

    def get_stock_OHLCV(self, symbol, start_year=None, end_year=None):
        OHLCV = fdr.DataReader(symbol=symbol, start=start_year, end=end_year)
        return OHLCV


if __name__ == "__main__":
    finance = FinanceData()
    stock_list = finance.get_stock_list()

    for stock_info in stock_list.to_numpy():
        symbol = stock_info[0]
        stock_info[2] = stock_info[2].replace("/", "")
        stock_info[2] = stock_info[2].replace(".", "")

        if len(symbol) != 6:
            pass

        elif symbol.isdigit() == False:
            pass

        else:
            OHLCV = finance.get_stock_OHLCV(symbol=symbol)

            if stock_info[1] == "KOSPI":
                # file_name = stock_info[0] + "-" + stock_info[1] + "-" + stock_info[2]
                file_name = stock_info[0]
                os.makedirs("/Users/macbook/Desktop/OHLCV_data/KOSPI_OHLCV", exist_ok=True)
                OHLCV.to_csv("/Users/macbook/Desktop/OHLCV_data/KOSPI_OHLCV/" + file_name)
            elif stock_info[1] == "KOSDAQ":
                # file_name = stock_info[0] + "-" + stock_info[1] + "-" + stock_info[2]
                file_name = stock_info[0]
                os.makedirs("/Users/macbook/Desktop/OHLCV_data/KOSDAQ_OHLCV", exist_ok=True)
                OHLCV.to_csv("/Users/macbook/Desktop/OHLCV_data/KOSDAQ_OHLCV/" + file_name)



