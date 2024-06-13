import matplotsoccer
import matplotlib.pyplot as plt

class Pitch:
    
    def __init__(self, color=None) -> None:
        self.color = color
        
    def __init_pitch(self):
        if self.color:
            f = matplotsoccer.field(show=False, color=self.color)
        else:
            f = matplotsoccer.field(show=False)

    def __show_player(self, x, y, color):
        plt.scatter(x, y, c=color,s=8)

        
    def show(self, players=None, corners=None):
        self.__init_pitch()
        if players:
            for p in players:
                self.__show_player(p.x+52.5, p.y+34, p.c)
        if corners:
            for c in corners:
                self.__show_player(c.x+52.5, c.y+34, "black")
        plt.show()

    def save(self, players=None, corners=None, path=""):
        self.__init_pitch()
        if players:
            for p in players:
                self.__show_player(p.x+52.5, p.y+34, p.c)
        if corners:
            for c in corners:
                self.__show_player(c.x+52.5, c.y+34, "black")
        plt.savefig(path)

if __name__ == "__main__":
    class Pls:
        def __init__(self, x, y, color):
            self.x, self.y, self.c = x, y, color
    p = Pitch("green")
    players = [
        Pls([10, 20], [20, 30], "red"),
        Pls([40+5, 40+5], [20+5, 30+5], "blue")
    ]
    corners = Pls([70+10, 70+10], [20+10, 30+10], "red")
    p.show(players, corners)